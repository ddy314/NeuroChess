// Fully native optional runtime. The locked baseline sources are included,
// never modified. No Python, LibTorch, callbacks, or runtime model conversion.
#include "mcts_pipeline.cpp"
#include "trt_inference.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now()-start).count();
}
using Output=std::array<float,515>;
struct Input {
  std::array<uint8_t,64> tokens;
  std::array<float,12> meta;
  std::array<int16_t,256> ids;
};
struct Miss { std::string key; Input input; std::vector<int> jobs; };
struct InferenceStats { uint64_t requested=0, inputs=0, hits=0, duplicates=0, batches=0; };
struct SearchResult { std::string move,json; uint64_t simulations=0; };

class Player {
  nc_native::TrtInference &network;
  Engine engine;
  size_t capacity;
  bool history_free;
  using CacheList=std::list<std::pair<std::string,Output>>;
  CacheList cache;
  std::unordered_map<std::string,CacheList::iterator> lookup;
  InferenceStats stats;
  std::vector<uint8_t> tokens;
  std::vector<float> meta;
  std::vector<int16_t> ids;
  std::vector<int> counts,jobids;
  std::vector<int> selected;
  std::vector<float> policy,q,values;

  std::string key(const Input &input) const {
    auto m=input.meta;
    if(history_free) for(int i:{5,6,7,10}) m[i]=0;
    std::string result(reinterpret_cast<const char *>(input.tokens.data()),64);
    result.append(reinterpret_cast<const char *>(m.data()),12*sizeof(float));
    result.append(reinterpret_cast<const char *>(input.ids.data()),256*sizeof(int16_t));
    return result;
  }
  void add(int job,const Output &output) {
    selected.push_back(job);
    policy.insert(policy.end(),output.begin(),output.begin()+256);
    q.insert(q.end(),output.begin()+256,output.begin()+512);
    values.push_back(output[512]);
  }
  void submit() {
    if(selected.empty()) return;
    engine.submit_some(selected.data(),policy.data(),q.data(),values.data(),selected.size());
    selected.clear(); policy.clear(); q.clear(); values.clear();
  }
  void store(const std::string &k,const Output &out) {
    if(!capacity) return;
    auto found=lookup.find(k);
    if(found!=lookup.end()) {
      found->second->second=out;
      cache.splice(cache.end(),cache,found->second);
    } else {
      cache.emplace_back(k,out); lookup.emplace(k,std::prev(cache.end()));
    }
    while(cache.size()>capacity) { lookup.erase(cache.front().first); cache.pop_front(); }
  }
public:
  Player(nc_native::TrtInference &net,int batch,size_t cache_size,bool history)
      :network(net),engine(Config{16384,batch,6,12,128,1.5f,.2f,1.5f,.75f}),
       capacity(cache_size),history_free(history),tokens(batch*64),meta(batch*12),
       ids(batch*256),counts(batch),jobids(batch) {
    engine.pending_loss_guard=true;
    selected.reserve(batch*8); policy.reserve(batch*8*256); q.reserve(batch*8*256); values.reserve(batch*8);
  }
  void clear() {
    engine.cancel_pending(); engine.trees.clear(); engine.cursor=0; engine.stats=Stats{};
    cache.clear(); lookup.clear(); stats=InferenceStats{};
  }
  SearchResult choose(const std::string &fen,const std::string &moves,double seconds,
                      int simulations,std::atomic<bool> &stop) {
    auto start=Clock::now(); auto previous=stats;
    engine.c.simulations=simulations;
    engine.cancel_pending();
    if(nc_resize(&engine,1)<0 || nc_start(&engine,0,fen.c_str(),moves.c_str())<0) {
      std::string reason=nc_error();clear();throw std::runtime_error(reason);
    }
    engine.time_fraction=seconds>0 ? 0 : -1;
    std::vector<Miss> pool; pool.reserve(engine.c.batch);
    double longest=0; bool complete=false,stopped=false;
    auto due=[&] {
      return nc_root_ready(&engine) && (stop.load() ||
        (seconds>0 && elapsed(start)>=seconds-std::max(.006,1.15*longest)));
    };
    try {
      while(!complete) {
        if(due()) { stopped=true; break; }
        auto round=Clock::now(); int added=0;
        while(int(pool.size())<engine.c.batch) {
          if(seconds>0) engine.time_fraction=std::min(1.,elapsed(start)/seconds);
          int n=engine.append(std::max(1,engine.c.batch-int(pool.size())),tokens.data(),
              meta.data(),ids.data(),counts.data(),jobids.data());
          if(!n) break;
          added+=n;
          for(int i=0;i<n;++i) {
            Input input;
            std::copy_n(tokens.data()+i*64,64,input.tokens.data());
            std::copy_n(meta.data()+i*12,12,input.meta.data());
            std::copy_n(ids.data()+i*256,256,input.ids.data());
            auto k=key(input); ++stats.requested;
            auto found=lookup.find(k);
            if(capacity && found!=lookup.end()) {
              add(jobids[i],found->second->second);
              cache.splice(cache.end(),cache,found->second); ++stats.hits;
              continue;
            }
            auto pending=capacity ? std::find_if(pool.begin(),pool.end(),
                [&](const Miss &p){return p.key==k;}) : pool.end();
            if(pending!=pool.end()) { pending->jobs.push_back(jobids[i]); ++stats.duplicates; }
            else pool.push_back(Miss{std::move(k),input,{jobids[i]}});
          }
          submit();
          if(added>=engine.c.batch*8 || due()) break;
        }
        if(pool.empty()) {
          if(!added) complete=true;
          longest=std::max(longest,elapsed(round)); continue;
        }
        if(due()) { stopped=true; break; }
        for(size_t i=0;i<pool.size();++i) {
          std::copy(pool[i].input.tokens.begin(),pool[i].input.tokens.end(),tokens.begin()+i*64);
          std::copy(pool[i].input.meta.begin(),pool[i].input.meta.end(),meta.begin()+i*12);
          std::copy(pool[i].input.ids.begin(),pool[i].input.ids.end(),ids.begin()+i*256);
        }
        auto output=network.evaluate(tokens.data(),meta.data(),ids.data(),pool.size());
        stats.inputs+=pool.size(); ++stats.batches;
        for(size_t i=0;i<pool.size();++i) {
          Output result; std::copy_n(output+i*515,515,result.data());
          store(pool[i].key,result);
          for(int job:pool[i].jobs) add(job,result);
        }
        submit(); pool.clear(); longest=std::max(longest,elapsed(round));
      }
      engine.cancel_pending();
    } catch(...) {
      engine.cancel_pending(); selected.clear();policy.clear();q.clear();values.clear();
      cache.clear();lookup.clear();throw;
    }
    std::string native=nc_result(&engine);
    auto marker=native.find("\"move\":\"");
    if(marker==std::string::npos) throw std::runtime_error("Missing root move");
    marker+=8; auto end=native.find('"',marker); std::string move=native.substr(marker,end-marker);
    std::ostringstream json;
    json<<std::setprecision(12)<<"{\"move\":\""<<move<<"\",\"seconds\":"<<elapsed(start)
        <<",\"network_inputs\":"<<stats.inputs-previous.inputs
        <<",\"requested_positions\":"<<stats.requested-previous.requested
        <<",\"cache_hits\":"<<stats.hits-previous.hits
        <<",\"batch_duplicates\":"<<stats.duplicates-previous.duplicates
        <<",\"physical_batches\":"<<stats.batches-previous.batches
        <<",\"time_stopped\":"<<(stopped?"true":"false")<<",\"native\":"<<native<<"}";
    return {move,json.str(),uint64_t(engine.trees[0]->done)};
  }
};

struct Options {
  std::string plan,fen,moves;
  double seconds=.5;
  int simulations=16384,batch=8;
  size_t cache=32768;
  bool history=false,uci=false,graphs=true;
};
Options parse(int argc,char **argv) {
  Options o;
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    auto value=[&] { if(i+1>=argc) throw std::runtime_error("Missing option value"); return std::string(argv[++i]); };
    if(arg=="--engine") o.plan=value();
    else if(arg=="--fen") o.fen=value();
    else if(arg=="--moves") o.moves=value();
    else if(arg=="--time") o.seconds=std::stod(value());
    else if(arg=="--simulations") o.simulations=std::stoi(value());
    else if(arg=="--batch") o.batch=std::stoi(value());
    else if(arg=="--cache") { auto n=std::stoll(value()); if(n<0) throw std::runtime_error("Negative cache size"); o.cache=n; }
    else if(arg=="--history-free") o.history=true;
    else if(arg=="--uci") o.uci=true;
    else if(arg=="--no-graphs") o.graphs=false;
    else throw std::runtime_error("Unknown option: "+arg);
  }
  if(o.plan.empty() || (!o.uci && o.fen.empty()) || o.batch<1 || o.batch>16 ||
      o.simulations<1 || !std::isfinite(o.seconds) || (o.seconds!= -1 && o.seconds<=0))
    throw std::runtime_error("Use --engine PLAN --fen FEN [--time .5 | --time -1 --simulations N], or --uci");
  return o;
}
void uci_loop(Player &player,Options options) {
  std::string fen=constants::STARTPOS,moves,line;
  std::atomic<bool> stop{false}; std::thread worker; std::mutex output;
  auto join=[&] { stop=true; if(worker.joinable()) worker.join(); };
  auto print=[&](const std::string &text) {std::lock_guard<std::mutex> lock(output);std::cout<<text<<std::endl;};
  while(std::getline(std::cin,line)) {
    std::istringstream in(line);std::string command;in>>command;
    try {
      if(command=="uci") print("id name NeuroChess Native CUDA experimental\nid author NeuroChess\nuciok");
      else if(command=="isready") print("readyok");
      else if(command=="quit") { join(); break; }
      else if(command=="stop") join();
      else if(command=="ucinewgame") {join();player.clear();}
      else if(command=="position") {
        join();std::string kind,word;in>>kind;moves.clear();
        if(kind=="startpos") fen=constants::STARTPOS;
        else if(kind=="fen") {
          fen.clear();for(int k=0;k<6;++k) {if(!(in>>word))throw std::runtime_error("Incomplete FEN");if(k)fen+=' ';fen+=word;}
        } else throw std::runtime_error("Unsupported position command");
        if(in>>word) {if(word!="moves")throw std::runtime_error("Expected moves");std::getline(in,moves);}
      } else if(command=="go") {
        join();double seconds=options.seconds;int nodes=options.simulations;bool movetime=false,fixed=false;
        std::string word;
        while(in>>word) {
          if(word=="movetime") {int ms;if(!(in>>ms)||ms<1)throw std::runtime_error("Invalid movetime");seconds=ms/1000.;movetime=true;}
          else if(word=="nodes") {if(!(in>>nodes)||nodes<1)throw std::runtime_error("Invalid nodes");fixed=true;}
          else if(word=="infinite") {seconds=-1;nodes=2147483647;}
          else if(word=="wtime" || word=="btime" || word=="winc" || word=="binc" || word=="movestogo") {in>>word;}
        }
        if(fixed && !movetime) seconds=-1;
        stop=false;
        worker=std::thread([&,initial=fen,history=moves,seconds,nodes] {
          try {
            auto result=player.choose(initial,history,seconds,nodes,stop);
            print("info nodes "+std::to_string(result.simulations)+" string telemetry "+result.json);
            print("bestmove "+result.move);
          } catch(const std::exception &e) {print(std::string("info string error ")+e.what());print("bestmove 0000");}
        });
      }
    } catch(const std::exception &e) {print(std::string("info string error ")+e.what());}
  }
  join();
}
} // namespace

int main(int argc,char **argv) {
  try {
    auto options=parse(argc,argv); auto start=Clock::now();
    nc_native::TrtInference network(options.plan,options.batch,options.graphs);
    std::cerr<<"native_setup_seconds "<<elapsed(start)<<'\n';
    Player player(network,options.batch,options.cache,options.history);
    if(options.uci) uci_loop(player,options);
    else {std::atomic<bool> stop{false};std::cout<<player.choose(options.fen,options.moves,options.seconds,options.simulations,stop).json<<'\n';}
    return 0;
  } catch(const std::exception &e) {std::cerr<<"native runtime: "<<e.what()<<'\n';return 1;}
}
