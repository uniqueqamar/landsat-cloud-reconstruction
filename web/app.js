"use strict";
const $=id=>document.getElementById(id);
let catalog,selected,image,mask,reference,prediction,session,loading=false,mode="rgb",selectionId=0;
const cache=new Map();
const N=128*128;
function status(text){$("status").textContent=text;}
function fail(error){$("error").hidden=false;$("error").textContent=error.message||String(error);}
async function read(url){const r=await fetch(url);if(!r.ok)throw new Error(`Could not load ${url} (${r.status}). Please retry.`);return r;}
function draw(canvas,values,overlay=false){
 const c=canvas.getContext("2d"),pixels=c.createImageData(128,128),bands=mode==="rgb"?[2,1,0]:[3,2,1];
 for(let i=0;i<N;i++){
  for(let k=0;k<3;k++)pixels.data[i*4+k]=Math.round(255*Math.pow(Math.max(0,Math.min(1,values[bands[k]*N+i]/.35)),1/1.8));
  if(overlay&&mask&&mask[i]>.5){pixels.data[i*4]=pixels.data[i*4]*.68+236*.32;pixels.data[i*4+1]=pixels.data[i*4+1]*.68+178*.32;pixels.data[i*4+2]=pixels.data[i*4+2]*.68+69*.32;}
  pixels.data[i*4+3]=255;
 }
 c.putImageData(pixels,0,0);
}
function redraw(){if(image)draw($("input"),image,$("mask").checked);if(prediction)draw($("output"),prediction,$("mask").checked);if(reference)draw($("reference"),reference);}
function resultMetric(){if(!prediction||!reference)return;let error=0,count=0;for(let i=0;i<N;i++){if(mask[i]>.5){for(let b=0;b<4;b++)error+=Math.abs(prediction[b*N+i]-reference[b*N+i]);count+=4;}}$("patchMetric").textContent=`This patch: ${(error/count).toFixed(5)} masked MAE against the later reference. This is a proxy comparison, not verified hidden-ground accuracy.`;}
function busy(value){loading=value;$("run").disabled=value||!image;document.querySelectorAll(".sample-card").forEach(b=>b.disabled=value);}
async function select(sample){
 if(loading)return;const id=++selectionId;selected=sample;image=null;prediction=null;reference=null;$("error").hidden=true;$("run").disabled=true;$("download").disabled=true;$("empty").hidden=false;$("outputBadge").textContent="NOT RUN";$("patchMetric").textContent="Run reconstruction to measure error against this reference.";
 document.querySelectorAll(".sample-card").forEach(b=>{const active=b.dataset.id===sample.id;b.classList.toggle("selected",active);b.setAttribute("aria-pressed",String(active));});
 $("patchTitle").textContent=sample.name;$("patchInfo").textContent=`${sample.coordinates} · ${Math.round(sample.mask_fraction*100)}% cloud / shadow mask`;status("Loading satellite patch…");
 try{
  const [input,target]=await Promise.all([read(sample.input).then(r=>r.arrayBuffer()),read(sample.reference).then(r=>r.arrayBuffer())]);
  if(id!==selectionId)return;
  const values=new Float32Array(input);if(values.length!==5*N||target.byteLength!==4*N*4)throw new Error("Patch dimensions are invalid.");
  image=values.slice(0,4*N);mask=values.slice(4*N);reference=new Float32Array(target);$("run").disabled=false;
  if(cache.has(sample.id)){prediction=cache.get(sample.id).values;$("empty").hidden=true;$("outputBadge").textContent="PREVIOUS RUN";$("download").disabled=false;resultMetric();status("Previous result on this device. Run again to refresh.");}else status("Ready. Model runs locally in your browser.");redraw();
 }catch(e){if(id===selectionId){fail(e);status("Patch could not be loaded. Select another patch or refresh.");}}
}
$("run").addEventListener("click",async()=>{
 if(loading||!image)return;busy(true);$("error").hidden=true;$("run").textContent="Reconstructing…";
 try{
  if(!session){status("Loading model and WebAssembly runtime (about 17 MB on first run)…");if(typeof ort==="undefined")throw new Error("The inference runtime did not load. Check your connection and refresh.");ort.env.wasm.wasmPaths=new URL("vendor/",location.href).href;ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;session=await ort.InferenceSession.create("assets/reconstruction.onnx",{executionProviders:["wasm"],graphOptimizationLevel:"all"});}
  status("Running reconstruction on this device…");await new Promise(requestAnimationFrame);
  const start=performance.now();const result=await session.run({image:new ort.Tensor("float32",image,[1,4,128,128]),mask:new ort.Tensor("float32",mask,[1,1,128,128])});const ms=performance.now()-start;
  prediction=Float32Array.from(result.prediction.data);if(prediction.length!==4*N||!prediction.every(Number.isFinite))throw new Error("Model returned invalid values.");
  cache.set(selected.id,{values:prediction,ms});$("empty").hidden=true;$("outputBadge").textContent="ESTIMATE";$("download").disabled=false;redraw();resultMetric();status(`Completed on this device · ${Math.round(ms)} ms inference · clear pixels preserved.`);
  window.lastInference={sample:selected.id,inferenceMs:ms,finite:true,length:prediction.length};
 }catch(e){fail(e);status("Reconstruction could not complete. You can retry; no prerecorded result is substituted.");}
 finally{busy(false);$("run").innerHTML='Run reconstruction <span>↗</span>';}
});
$("rgb").addEventListener("click",()=>{mode="rgb";$("rgb").classList.add("active");$("nir").classList.remove("active");$("rgb").setAttribute("aria-pressed","true");$("nir").setAttribute("aria-pressed","false");redraw();});
$("nir").addEventListener("click",()=>{mode="nir";$("nir").classList.add("active");$("rgb").classList.remove("active");$("nir").setAttribute("aria-pressed","true");$("rgb").setAttribute("aria-pressed","false");redraw();});
$("mask").addEventListener("change",redraw);
$("download").addEventListener("click",()=>{if(!prediction)return;const canvas=document.createElement("canvas");canvas.width=canvas.height=128;draw(canvas,prediction);canvas.toBlob(blob=>{const u=URL.createObjectURL(blob),a=document.createElement("a");a.href=u;a.download=`${selected.id}-estimated-${mode}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);});});
async function init(){try{catalog=await read("assets/samples.json").then(r=>r.json());$("sampleCount").textContent=String(catalog.samples.length).padStart(2,"0");for(const sample of catalog.samples){const b=document.createElement("button");b.className="sample-card";b.dataset.id=sample.id;b.setAttribute("aria-pressed","false");const img=document.createElement("img");img.src=sample.thumbnail;img.alt="";const info=document.createElement("span"),name=document.createElement("strong"),detail=document.createElement("small");name.textContent=sample.name;detail.textContent=`${Math.round(sample.cloud_fraction*100)}% cloud`;info.append(name,detail);b.append(img,info);b.addEventListener("click",()=>select(sample));$("samples").append(b);}await select(catalog.samples[0]);}catch(e){fail(e);status("Could not load dataset. Please refresh to retry.");}}
init();
