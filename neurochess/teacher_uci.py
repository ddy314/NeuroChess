"""Resident synchronous Stockfish transport for CPU label workers.

Keep only the last info line for each MultiPV index. Intermediate principal
variations are never converted to Python chess Boards/Moves. Every search still
uses ucinewgame+isready, preserving independent teacher-budget checks.
"""
import os
import re
import select
import subprocess
import time
import chess
import chess.engine


class FastTeacher:
    def __init__(self, binary, hash_mb=64, threads=1, timeout=180):
        self.timeout=timeout;self.buffer=b'';self.closed=False;self.multipv=1
        self.process=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL,bufsize=0)
        self.stats=dict(searches=0,info_lines=0,seconds=0.,reset_seconds=0.)
        try:
            self.send('uci');self.identity=[]
            for line in self.until('uciok'):
                if line.startswith('id '):self.identity.append(line)
            self.send(f'setoption name Threads value {threads}\nsetoption name Hash value {hash_mb}\n'
                      'setoption name Skill Level value 20\nsetoption name UCI_LimitStrength value false\n'
                      'setoption name UCI_Chess960 value false\nsetoption name UCI_ShowWDL value false\nisready')
            list(self.until('readyok'))
        except BaseException:
            self.quit();raise

    def send(self,command):
        if self.closed:raise RuntimeError('Teacher is closed')
        self.process.stdin.write((command+'\n').encode())

    def until(self,prefix):
        deadline=time.monotonic()+self.timeout
        while True:
            if b'\n' not in self.buffer:
                remaining=deadline-time.monotonic()
                if remaining<=0 or not select.select([self.process.stdout],[],[],remaining)[0]:
                    self.quit();raise TimeoutError(f'Stockfish did not return {prefix}')
                block=os.read(self.process.stdout.fileno(),65536)
                if not block:
                    self.quit();raise RuntimeError('Stockfish exited before '+prefix)
                self.buffer+=block
                continue
            line,self.buffer=self.buffer.split(b'\n',1);line=line.decode('ascii',errors='replace').strip()
            if line.startswith(prefix):return
            yield line

    def analyse(self,board,limit,multipv,game=None,root_moves=None):
        if not limit.nodes or multipv<1:raise ValueError('Positive fixed nodes and MultiPV required')
        start=time.monotonic()
        if multipv!=self.multipv:
            self.send(f'setoption name MultiPV value {multipv}');self.multipv=multipv
        self.send('ucinewgame\nisready');list(self.until('readyok'))
        self.stats['reset_seconds']+=time.monotonic()-start
        root=board.root();fen=root.fen(en_passant='fen')
        command='position '+('startpos' if fen==chess.STARTING_FEN else 'fen '+fen)
        if board.move_stack:command+=' moves '+' '.join(m.uci() for m in board.move_stack)
        self.send(command)
        command=f'go nodes {int(limit.nodes)}'
        if root_moves is not None:
            if not root_moves:raise ValueError('Empty restricted candidate set')
            command+=' searchmoves '+' '.join(m.uci() for m in root_moves)
        self.send(command);latest={}
        for line in self.until('bestmove '):
            if not line.startswith('info ') or ' score ' not in line or ' pv ' not in line:continue
            self.stats['info_lines']+=1
            match=re.search(r'\bmultipv (\d+)\b',line)
            latest[int(match[1]) if match else 1]=line
        result=[]
        for _,line in sorted(latest.items()):
            head,pv=line.split(' pv ',1);tokens=head.split();get=lambda name:int(tokens[tokens.index(name)+1])
            i=tokens.index('score');kind=tokens[i+1];score=int(tokens[i+2])
            if kind not in ('cp','mate'):raise ValueError('Unknown UCI score '+kind)
            info=dict(pv=[chess.Move.from_uci(m) for m in pv.split()],
                      score=chess.engine.PovScore(chess.engine.Mate(score) if kind=='mate' else chess.engine.Cp(score),board.turn),
                      depth=get('depth'),nodes=get('nodes'))
            if 'lowerbound' in tokens:info['lowerbound']=True
            if 'upperbound' in tokens:info['upperbound']=True
            result.append(info)
        if not result:raise RuntimeError('No scored legal principal variation returned')
        self.stats['searches']+=1;self.stats['seconds']+=time.monotonic()-start
        return result

    def quit(self):
        if self.closed:return
        self.closed=True
        try:
            self.process.stdin.write(b'quit\n');self.process.wait(timeout=3)
        except (BrokenPipeError,subprocess.TimeoutExpired):
            self.process.kill();self.process.wait()
        finally:
            self.process.stdin.close();self.process.stdout.close()
