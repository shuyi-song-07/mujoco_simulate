import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Exercise the actual app's transport seam, not just its packet constructor.
const source=fs.readFileSync(new URL('./single-arm-app.js',import.meta.url),'utf8');
const pumpSource=source.slice(source.indexOf('async function pump()'),source.indexOf('\nfunction pause('));
async function dispatch(input,operation=null,now=100060) {
  const sent=[];
  const context={packetFlight:false,connected:true,shutdown:false,eventQueue:[],latestInput:input,
    runtime:{session:'s',control_epoch:0},engine:{parameters:{maximum_age_ms:150}},operation,
    Date:{now:()=>now},journal:{add(){}},context:()=>({}),logSingleControlResult(){},
    stamped:p=>({...p,eventId:p.eventId??'new-event'}),
    jsonRequest:async(_,p)=>{sent.push(p);return {ok:true,queued:true};},render(){}};
  vm.runInNewContext(pumpSource+'; this.run=pump;',context);
  await context.run();return sent;
}
test('continuous input preserves capture time with no pending button operation',async()=>{
  const input={command:'continuous_input',sentAt:100000,measurement_id:'m',session:'s',control_epoch:0};
  const sent=await dispatch(input);
  assert.equal(sent.length,1);assert.equal(sent[0].sentAt,100000);
  assert.equal(input.sentAt,100000);
});
test('an expired camera frame is dropped, never refreshed into a fresh frame',async()=>{
  assert.equal((await dispatch({command:'continuous_input',sentAt:100000},null,100151)).length,0);
});
test('only an explicit matching button operation or pause gets dispatch time',async()=>{
  const op={command:'record_save',eventId:'save-1'};
  assert.equal((await dispatch({...op,sentAt:90000},op))[0].sentAt,100060);
  assert.equal((await dispatch({command:'pause',sentAt:90000}))[0].sentAt,100060);
  assert.equal((await dispatch({command:'continuous_input',sentAt:100000},op))[0].sentAt,100000);
});
