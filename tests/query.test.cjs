const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function setup() {
  const elements = new Map();
  const node = selector => {
    if (!elements.has(selector)) elements.set(selector, {value:'',textContent:'',dataset:{},querySelector:s=>node(s),querySelectorAll:()=>[]});
    return elements.get(selector);
  };
  const context = vm.createContext({URLSearchParams, AbortController, console, setTimeout,
    document:{addEventListener(){},querySelector:node,querySelectorAll:()=>[]}});
  for (const file of ['app.js','query.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'../web',file),'utf8'),context);
  vm.runInContext('renderLineChart=()=>{};renderLegend=()=>{};toast=()=>{};for(const d of Object.values(queryDefinitions))d.render=()=>{};',context);
  node('#resultObject').value='OBJ-GRID';node('#resultHorizon').value='week_ahead';node('#resultMetric').value='all';
  node('.query-point').value='P002';node('#weatherResultHorizon').value='hour_ahead';node('#weatherResultMetric').value='all';
  return {context,node};
}

test('weather export stays on its own query after another page refreshes', async()=>{
  const {context,node}=setup();
  context.fetch=async url=>({ok:true,json:async()=>({run_id:url.includes('kind=weather')?'WEATHER':'POWER', rows:[{timestamp:'2026-01-01T00:00:00Z',wind_speed_mps:6,pv_power_kw:99}]})});
  await vm.runInContext("queryPage('weather-results')",context);
  await vm.runInContext("queryPage('results')",context);
  node('.view.active').dataset.page='weather-results';
  vm.runInContext("state.currentRows=[{timestamp:'wrong'}];downloadCSV=(rows,columns,name)=>{globalThis.exported={rows,columns,name}};exportCurrentRows()",context);
  assert.equal(context.exported.rows[0].run_id,'WEATHER');
  assert.equal(context.exported.rows[0].point_id,'P002');
  assert.ok(context.exported.columns.includes('wind_speed_mps'));
});

test('slower old query cannot overwrite latest selection or export data', async()=>{
  const {context}=setup();const responses=[];
  context.fetch=()=>new Promise(resolve=>responses.push(resolve));
  const first=vm.runInContext("queryPage('results')",context);
  const second=vm.runInContext("queryPage('results')",context);
  responses[1]({ok:true,json:async()=>({run_id:'LATEST',rows:[{pv_power_kw:2}]})});await second;
  responses[0]({ok:true,json:async()=>({run_id:'OLD',rows:[{pv_power_kw:1}]})});await first;
  assert.equal(vm.runInContext("querySnapshots.get('results').runId",context),'LATEST');
});

test('failed queries do not replace missing observations with sample curves', async()=>{
  const {context}=setup();
  context.fetch=async()=>({ok:false,json:async()=>({error:'offline'})});
  await vm.runInContext("queryPage('weather-results')",context);
  assert.equal(vm.runInContext("querySnapshots.get('weather-results').rows.length",context),0);
  assert.equal(vm.runInContext("querySnapshots.get('weather-results').error",context),'offline');
});

test('energy quality uses the same time range and clears stale values on failure', async()=>{
  const {context,node}=setup();
  vm.runInContext("queryDefinitions.energy.render=renderQuality",context);
  context.fetch=async()=>({ok:true,json:async()=>({measurements:[
    {timestamp:new Date().toISOString(),quality_code:'valid',curtailment_flag:1,pv_curtailment_kw:4},
    {timestamp:'2000-01-01T00:00:00Z',quality_code:'valid',pv_curtailment_kw:100}
  ]})});
  await vm.runInContext("queryPage('energy')",context);
  assert.equal(vm.runInContext("querySnapshots.get('energy-input').rows.length",context),1);
  assert.match(node('#qualityGrid').innerHTML,/<b>1.0 kWh<\/b>/);
  context.fetch=async()=>({ok:false,json:async()=>({error:'offline'})});
  await vm.runInContext("queryPage('energy')",context);
  assert.match(node('#qualityGrid').innerHTML,/<span>采集记录<\/span><b>0<\/b>/);
  assert.doesNotMatch(node('#qualityGrid').innerHTML,/1.0 kWh/);
});
