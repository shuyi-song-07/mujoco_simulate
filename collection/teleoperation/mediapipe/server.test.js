import test from "node:test";
import assert from "node:assert/strict";
import {createServer} from "node:http";
import {spawn} from "node:child_process";
import {fileURLToPath} from "node:url";

async function launch(t) {
  const reservation=createServer();await new Promise(resolve=>reservation.listen(0,"127.0.0.1",resolve));
  const port=reservation.address().port;await new Promise(resolve=>reservation.close(resolve));
  const child=spawn(process.execPath,[fileURLToPath(new URL("./server.js",import.meta.url))],{
    env:{...process.env,PORT:String(port)},windowsHide:true,stdio:["ignore","pipe","pipe"]});
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
