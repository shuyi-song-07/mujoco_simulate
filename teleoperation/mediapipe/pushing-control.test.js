import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {mapPlanarMotion} from './interaction-state.js';

test('Push-T palm/fist re-anchor Z motion and planar entry locks the current height', () => {
  const code=readFileSync(new URL('./app.js',import.meta.url),'utf8');
  const source=code.slice(code.indexOf('function processPushingControl('),code.indexOf('function setCameraLoading('));
  const requests=[];
  const context={robotFeedback:{},lastPlanarPoint:null,pushingPlanarGestureStartedAt:-Infinity,
    pendingMotionRequest:undefined,lastOpenPalmY:null,robotZLocked:true,lastControlSentAt:0,
    CONTROL_INTERVAL_MS:50,verticalDeadZone:()=>.015,planarSensitivity:()=>.03,
    planarDeadZone:()=>.09,mapPlanarMotion,
    sendRobotCommand:(command)=>requests.push(command),
    sendLatestMotionCommand:(command)=>requests.push(command)};
  vm.createContext(context);vm.runInContext(source,context);
  const send=(gesture,y,timestamp,gestureChanged)=>context.processPushingControl({
    gesture,wrist:{x:.5,y},timestamp,gestureChanged});
  send('Open_Palm',.5,100,true);
  assert.deepEqual(requests,['start_vertical']);assert.equal(context.robotZLocked,false);
  send('Open_Palm',.45,200,false);assert.equal(requests.at(-1),'up');
  send('Closed_Fist',.8,300,true);assert.equal(requests.at(-1),'start_vertical');
  send('Closed_Fist',.84,400,false);assert.equal(requests.at(-1),'down');
  send('ILoveYou',.84,500,true);assert.equal(requests.at(-1),'start_planar');
  assert.equal(context.robotZLocked,true);
  assert.ok(!requests.includes('open')&&!requests.includes('close'));
});
