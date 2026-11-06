"""Original full training dashboard plus the live local annotation counters."""
import argparse
import json
import time
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from neurochess.dashboard import WEB, read_json
from scripts.targeted_full_dashboard import Handler as FullHandler, full_snapshot, PANEL, EXTRA


LOCAL_JS = '''
const previousLocalRender=render;
render=function(){previousLocalRender();const p=data.local_annotation;if(!p)return;
 const n=x=>Number(x||0).toLocaleString(),c=p.labelled||{};
 const stale=data.server_time-(p.updated_at||0)>180;
 txt('phase',p.phase==='annotation-complete-training-deferred'?'标注完成，等待训练':p.phase==='failed-local-annotation'?'本地标注失败':stale?'本地标注日志未更新':'本地标注中');
 txt('screening-info',`合格标注：训练 ${n(c.train)} / 500,000 · 验证 ${n(c.val)} / 3,000 · 测试 ${n(c.test)} / 3,000；已筛 ${n(p.screened)} / 100,000 · 确认训练错题 ${n(p.verified_training_mistakes)} / 10,000 · 不稳定标签 ${n(p.unstable_discarded)}`);
 txt('notice',`${p.cloud_released?'云端 L4 已释放':'云端释放待核实'} · 本地 ${n(p.workers)} 路 Stockfish · 本次处理 ${n(p.local_completed)} 局面 · 平均 ${Number(p.local_positions_per_second||0).toFixed(1)} 局面/秒。下方曲线是已完成的 2000 步试训历史，当前没有训练。`);
 if(stale)txt('notice','标注状态已超过 3 分钟未更新，请检查本地进程。下方为历史训练曲线。');
 txt('source','来源：本地 pipeline-status.json、SQLite 逐题检查点；历史曲线来自迁移保存的云端试训日志。');
 txt('score-note','试训后的搜索评测未完成；没有新的棋力结论');
};
'''


class Handler(FullHandler):
    def do_GET(self):
        path=urlparse(self.path).path
        if path in ['/','/index.html']:
            destination=read_json(self.root/'pipeline-status.json').get('training_dashboard')
            if destination=='http://127.0.0.1:18767/':
                self.send_response(302);self.send_header('Location',destination);self.end_headers();return
        if path=='/api/run':
            d=full_snapshot(self.root/'pilot-history')
            d['local_annotation']=read_json(self.root/'pipeline-status.json')
            d['server_time']=time.time()
            return self.send(json.dumps(d).encode())
        if path=='/api/progress':
            return self.send(json.dumps(read_json(self.root/'pipeline-status.json')).encode())
        if path in ['/','/index.html']:
            panel=PANEL.replace('已筛局面包含双方后续4个半回合','新精英棋谱实际局面 · 完整历史')
            panel=panel.replace('确认错题按训练／验证／测试分开；争议标签不进入错题训练。完整训练窗口为2000步，阶段验证不自动晋升模型。',
                                '本地仅标注，不训练。争议标签不入训练；旧试训曲线保留供查阅。')
            html=(WEB/'index.html').read_text().replace('<section class="panel">',panel+'<section class="panel">',1)
            html=html.replace('<h2>Q 预测误差</h2>','<h2>通用 Value 预测误差</h2>').replace('仅已标注候选 · MAE','历史独立通用验证集 · MAE')
            return self.send(html.encode(),'text/html; charset=utf-8')
        if path=='/app.js':
            return self.send(((WEB/'app.js').read_text()+EXTRA+LOCAL_JS).encode(),'text/javascript; charset=utf-8')
        return super().do_GET()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    ThreadingHTTPServer(('127.0.0.1',a.port),partial(Handler,root=a.root)).serve_forever()
