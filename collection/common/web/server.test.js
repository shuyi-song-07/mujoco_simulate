import test from "node:test";
import assert from "node:assert/strict";
import {createServer} from "node:http";
import {spawn} from "node:child_process";
import {fileURLToPath} from "node:url";

async function launch(t, backendPort) {
  const reservation=createServer();await new Promise(resolve=>reservation.listen(0,"127.0.0.1",resolve));
  const port=reservation.address().port;await new Promise(resolve=>reservation.close(resolve));
  const child=spawn(process.execPath,[fileURLToPath(new URL("./server.js",import.meta.url))],{
    env:{...process.env,PORT:String(port),...(backendPort ? {DUAL_BACKEND_PORT:String(backendPort)} : {})},windowsHide:true,stdio:["ignore","pipe","pipe"]});
  t.after(()=>child.kill());
  await new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error("Static server did not become ready")),5000);
    child.once("error",error=>{clearTimeout(timer);reject(error);});
    child.once("exit",code=>{clearTimeout(timer);reject(new Error(`Static server exited ${code}`));});
    child.stdout.on("data",data=>{if(String(data).includes("Gesture Interaction is running")){clearTimeout(timer);resolve();}});
  });
  return `http://127.0.0.1:${port}`;
}
test("the real static service serves the full browser module graph with JavaScript MIME types",async(t)=>{
  const base=await launch(t),pending=["/dual-arm-app.js","/dual-gesture-worker.js"],visited=new Set();
  while(pending.length) {
    const path=pending.shift();if(visited.has(path))continue;visited.add(path);
    const response=await fetch(base+path);
    assert.equal(response.status,200,`${path} must be available to browser imports`);
    assert.match(response.headers.get("content-type"),/^text\/javascript/);
    const source=await response.text();
    for(const match of source.matchAll(/\bfrom\s+["'](\.[^"']+)["']/g))pending.push(new URL(match[1],base+path).pathname);
  }
  assert.ok(visited.has("/rate-interaction-state.js"));
  assert.ok(visited.has("/continuous-interaction-state.js"));
  assert.ok(visited.has("/hand-measurement.js"));
  for(const path of ["/server.js","/package.json","/rate-interaction-state.test.js"]) {
    assert.equal((await fetch(base+path)).status,404,`private source remains unserved: ${path}`);
  }
});

test("three simulation routes retain freshness headers and oblique is the only slanted view",async(t)=>{
  const backend=createServer((request,response)=>{
    assert.match(request.url,/^\/(top|front|oblique)-preview$/);
    response.writeHead(200,{"content-type":"image/jpeg","x-preview-age-ms":"125","x-preview-epoch":"4"});
    response.end(request.url);
  });
  await new Promise(resolve=>backend.listen(0,"127.0.0.1",resolve));
  t.after(()=>backend.close());
  const base=await launch(t,backend.address().port);
  for(const camera of ["top","front","oblique"]) {
    const response=await fetch(`${base}/api/dual/${camera}-preview`);
    assert.equal(response.status,200);
    assert.equal(response.headers.get("content-type"),"image/jpeg");
    assert.equal(response.headers.get("x-preview-age-ms"),"125");
    assert.equal(response.headers.get("x-preview-epoch"),"4");
    assert.equal(await response.text(),`/${camera}-preview`);
  }
  const page=await (await fetch(`${base}/dual_arm.html`)).text();
  assert.doesNotMatch(page,/id="auxiliary-preview-three"/);
  assert.doesNotMatch(page,/option value="side"/);
  assert.match(page,/option value="oblique"/);
  assert.equal((await fetch(`${base}/api/dual/side-preview`)).status,404);
  assert.match(page,/<video id="webcam"/);
});
