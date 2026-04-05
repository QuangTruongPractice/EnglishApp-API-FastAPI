"""Giao diện web thu âm → WAV. Chạy: python recorder.py → mở http://localhost:9000"""

import os, io, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydub import AudioSegment

app = FastAPI()
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_audio")
os.makedirs(OUT_DIR, exist_ok=True)

HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audio Recorder</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{min-height:100vh;display:flex;align-items:center;justify-content:center;
  background:#0f0f1a;font-family:'Segoe UI',system-ui,sans-serif;color:#e0e0e0}
.card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);
  border-radius:24px;padding:48px 40px;text-align:center;width:420px;
  backdrop-filter:blur(20px);box-shadow:0 20px 60px rgba(0,0,0,.4)}
h1{font-size:1.6rem;margin-bottom:8px;background:linear-gradient(135deg,#6EE7B7,#3B82F6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub{font-size:.85rem;color:#888;margin-bottom:32px}
.btn{width:160px;height:160px;border-radius:50%;border:none;cursor:pointer;
  font-size:1rem;font-weight:600;transition:all .3s;position:relative;outline:none}
.btn.idle{background:linear-gradient(135deg,#3B82F6,#6366F1);color:#fff;
  box-shadow:0 8px 30px rgba(99,102,241,.4)}
.btn.idle:hover{transform:scale(1.05);box-shadow:0 12px 40px rgba(99,102,241,.6)}
.btn.rec{background:linear-gradient(135deg,#EF4444,#F97316);color:#fff;
  box-shadow:0 8px 30px rgba(239,68,68,.5);animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,.4)}
  50%{box-shadow:0 0 0 20px rgba(239,68,68,0)}}
.timer{font-size:2.4rem;font-weight:700;margin:24px 0 8px;font-variant-numeric:tabular-nums;
  letter-spacing:2px}
.status{font-size:.85rem;color:#888;min-height:20px;margin-bottom:16px}
.filename{margin-top:16px}
.filename input{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.15);
  border-radius:12px;padding:10px 16px;color:#e0e0e0;font-size:.9rem;width:100%;
  text-align:center;outline:none}
.filename input:focus{border-color:#6366F1}
.result{margin-top:20px;padding:14px;border-radius:12px;font-size:.85rem;display:none}
.result.ok{background:rgba(16,185,129,.15);color:#6EE7B7;display:block}
.result.err{background:rgba(239,68,68,.15);color:#F87171;display:block}
audio{margin-top:16px;width:100%;border-radius:8px}
</style>
</head>
<body>
<div class="card">
  <h1>🎙 Audio Recorder</h1>
  <p class="sub">Thu âm và lưu thành file WAV</p>
  <div class="filename">
    <input id="fname" type="text" placeholder="Tên file (VD: hello)" value="">
  </div>
  <div class="timer" id="timer">00:00</div>
  <p class="status" id="status">Nhấn để bắt đầu thu âm</p>
  <button class="btn idle" id="btn" onclick="toggle()">BẮT ĐẦU</button>
  <div class="result" id="result"></div>
  <audio id="player" controls style="display:none"></audio>
</div>
<script>
let mediaRec,chunks=[],recording=false,start,timerI;
const btn=document.getElementById('btn'),timer=document.getElementById('timer'),
      status=document.getElementById('status'),result=document.getElementById('result'),
      player=document.getElementById('player'),fname=document.getElementById('fname');

function fmt(s){const m=Math.floor(s/60),ss=Math.floor(s%60);
  return String(m).padStart(2,'0')+':'+String(ss).padStart(2,'0')}

async function toggle(){
  if(!recording){
    try{
      const stream=await navigator.mediaDevices.getUserMedia({audio:true});
      mediaRec=new MediaRecorder(stream,{mimeType:'audio/webm;codecs=opus'});
      chunks=[];
      mediaRec.ondataavailable=e=>{if(e.data.size>0)chunks.push(e.data)};
      mediaRec.onstop=async()=>{
        stream.getTracks().forEach(t=>t.stop());
        const blob=new Blob(chunks,{type:'audio/webm'});
        player.src=URL.createObjectURL(blob);player.style.display='block';
        status.textContent='Đang chuyển đổi…';
        const fd=new FormData();
        fd.append('audio',blob,'rec.webm');
        fd.append('filename',fname.value.trim()||'recording');
        try{
          const r=await fetch('/save',{method:'POST',body:fd});
          const j=await r.json();
          if(j.ok){result.className='result ok';result.textContent='✓ Đã lưu: '+j.path}
          else{result.className='result err';result.textContent='✗ '+j.error}
        }catch(e){result.className='result err';result.textContent='✗ '+e.message}
        status.textContent='Nhấn để thu âm lại';
      };
      mediaRec.start();recording=true;start=Date.now();
      btn.textContent='DỪNG';btn.className='btn rec';
      status.textContent='Đang thu âm…';result.className='result';
      player.style.display='none';
      timerI=setInterval(()=>timer.textContent=fmt((Date.now()-start)/1000),200);
    }catch(e){status.textContent='Lỗi mic: '+e.message}
  }else{
    mediaRec.stop();recording=false;clearInterval(timerI);
    btn.textContent='BẮT ĐẦU';btn.className='btn idle';
  }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def page():
    return HTML


@app.post("/save")
async def save(request: Request):
    form = await request.form()
    audio = form["audio"]
    name = form.get("filename", "recording")
    name = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).strip() or "recording"
    try:
        raw = await audio.read()
        seg = AudioSegment.from_file(io.BytesIO(raw), format="webm")
        seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        out = os.path.join(OUT_DIR, f"{name}.wav")
        seg.export(out, format="wav")
        return JSONResponse({"ok": True, "path": out})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


if __name__ == "__main__":
    print(f"[INFO] Recordings -> {OUT_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=9000)
