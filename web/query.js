// Each query owns its data and request token. Background dashboard refreshes
// cannot replace the rows being displayed or exported on another page.
const querySnapshots = new Map();
const pinnedRuns = new Map();
const queryDefinitions = {
  results: {page:'results', chart:'#resultQueryChart', legend:'#resultLegend', hint:'#resultQueryHint', metric:'#resultMetric', horizon:'#resultHorizon', render:renderResultTable},
  'weather-results': {page:'weather-results', chart:'#weatherResultChart', legend:'#weatherResultLegend', hint:'#weatherResultHint', metric:'#weatherResultMetric', horizon:'#weatherResultHorizon', render:renderWeatherResultTable},
  weather: {page:'weather-input', chart:'#weatherQueryChart', legend:'#weatherLegend', hint:'#weatherQueryHint', metric:'#weatherMetric', render:renderWeatherTable},
  energy: {page:'energy-input', chart:'#energyQueryChart', legend:'#energyLegend', hint:'#energyQueryHint', metric:'#energyMetric', render:renderQuality}
};

function downloadCSV(rows, columns, name) {
  const labels={timestamp:'时间',wind_speed_mps:'风速(m/s)',temperature_c:'温度(°C)',solar_radiation_wm2:'太阳辐射(W/m²)',pv_power_kw:'光伏(kW)',wind_power_kw:'风电(kW)',load_kw:'负荷(kW)',source:'来源',point_id:'点位',object_id:'对象',run_id:'运行编号',horizon:'周期',metric:'变量',unit:'单位',min:'最小值',min_time:'最小值时刻',max:'最大值',max_time:'最大值时刻',mean:'平均值',count:'有效样本',missing:'缺测样本',integral:'累计量',integral_unit:'累计量单位',gaps:'时间缺口',duplicates:'重复时刻',date:'日期',time_zone:'时区'};
  const csv='\uFEFF'+[columns.map(key=>labels[key]||key),...rows.map(row=>columns.map(key=>row[key]))].map(row=>row.map(csvCell).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  const link=document.createElement('a');
  link.href=url;link.download=name+'.csv';link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}

async function queryPage(kind) {
  const def=queryDefinitions[kind], view=$('.view[data-page="'+def.page+'"]');
  const objectId=kind==='results'?$('#resultObject').value:null;
  const point=kind==='results'?objectById(objectId)?.point_id||state.selectedPoint:$('.query-point',view)?.value||state.selectedPoint;
  const horizon=def.horizon?$(def.horizon).value:null, metric=$(def.metric).value;
  let pin=pinnedRuns.get(def.page);
  if(!pin){
    const runId=typeof location!=='undefined'?new URLSearchParams(location.search).get('run_id'):null;
    if(runId)pin={run_id:runId};
  }
  if(pin && (pin.point_id!==point || pin.horizon!==horizon || (kind==='results'&&pin.object_id!==objectId))){
    pinnedRuns.delete(def.page);pin=null;
  }
  const previous=querySnapshots.get(def.page);
  previous?.controller?.abort();
  const snapshot={point,objectId,horizon,runId:pin?.run_id||'',rows:[],loading:true,error:null,controller:new AbortController()};
  querySnapshots.set(def.page,snapshot);
  $all('.export-btn',view).forEach(button=>button.disabled=true);
  $(def.hint).textContent='查询中…';
  const specs=metricSpecs(kind==='weather-results'?'weather':kind,metric);
  renderQueryChart($(def.chart),[],specs);
  def.render([]);
  renderQueryAnalytics(kind,snapshot);
  try{
    let path,field='rows';
    const hours=Number($('.query-range',view)?.value||24);
    if(horizon){
      const params=new URLSearchParams(pin?{run_id:pin.run_id}:kind==='results'?{object_id:objectId,horizon,kind:'energy'}:{point_id:point,horizon,kind:'weather'});
      path='/api/forecast?'+params;
    }else{
      field=kind==='weather'?'observations':'measurements';
      const end=new Date(),start=new Date(end.getTime()-hours*3600000);
      path='/api/'+(kind==='weather'?'weather-observations':'energy-measurements')+'?'+new URLSearchParams({point_id:point,limit:String(hours*4),start:start.toISOString(),end:end.toISOString()});
    }
    const response=await fetch(path,{signal:snapshot.controller.signal});
    const data=await response.json();
    if(!response.ok)throw new Error(data.error||'查询失败');
    if(querySnapshots.get(def.page)!==snapshot)return;
    snapshot.rows=(data[field]||[]).map(row=>({...row,timestamp:row.timestamp||row.timestamp_utc}));
    if(!horizon)snapshot.rows=snapshot.rows.filter(row=>Date.parse(row.timestamp)>=Date.now()-hours*3600000).sort((a,b)=>Date.parse(a.timestamp)-Date.parse(b.timestamp));
    snapshot.runId=data.run_id||snapshot.runId;
    snapshot.run=data.run||snapshot.run;
    if(horizon)snapshot.objectId=data.object_id||snapshot.objectId;
    snapshot.loading=false;
    renderQueryChart($(def.chart),snapshot.rows,specs);
    renderLegend(def.legend,specs);
    def.render(snapshot.rows);
    renderQueryAnalytics(kind,snapshot);
    const label=kind==='results'?objectById(objectId)?.object_name||objectId:pointById(point)?.name||point;
    $(def.hint).textContent=label+' · '+(horizon?horizonNames[horizon]:'最近 '+hours+' 小时实测')+' · '+snapshot.rows.length+' 条'+(snapshot.runId?' · '+snapshot.runId:'')+(snapshot.rows.length?'':' · 暂无数据');
    $all('.export-btn',view).forEach(button=>button.disabled=!snapshot.rows.length);
  }catch(error){
    if(querySnapshots.get(def.page)!==snapshot)return;
    snapshot.loading=false;snapshot.error=error.message;
    snapshot.run=null;
    $(def.hint).textContent='查询失败：'+error.message;
    renderQueryAnalytics(kind,snapshot);
  }
  const latest=$('[data-latest-results]',view);
  if(latest)latest.hidden=!pinnedRuns.has(def.page)&&!(typeof location!=='undefined'&&new URLSearchParams(location.search).has('run_id'));
}

function openRunResults(run) {
  if(run.status!=='success'){toast(run.error_message||'该运行尚无成功结果',true);return}
  const weather=run.forecast_kind==='weather', page=weather?'weather-results':'results';
  state.selectedPoint=run.point_id;
  window.location.href=`${pageUrls[page]||'index.html'}?run_id=${encodeURIComponent(run.run_id)}`;
}

// Page all rows inside the available panel space. Only one slice is visible;
// no row is discarded and no scrollbar is needed to reach the last sample.
function installTablePager(body) {
  const wrap=body.closest('.table-wrap'), panel=wrap?.parentElement;
  if(!wrap||!panel)return;
  panel.classList.add('paged-panel');
  const pager=document.createElement('div');
  pager.className='table-pager';
  const label=document.createElement('span');
  label.setAttribute('aria-live','polite');
  const previous=document.createElement('button'), next=document.createElement('button'), last=document.createElement('button');
  previous.textContent='上一页';next.textContent='下一页';last.textContent='末页';
  [previous,next,last].forEach(button=>{button.type='button';button.className='secondary-button'});
  const jump=document.createElement('input');jump.type='number';jump.min='1';jump.setAttribute('aria-label','跳转页码');
  pager.append(label,previous,jump,next,last);wrap.after(pager);
  let page=0,pages=1;
  function draw(){
    if(!wrap.getBoundingClientRect().height)return;
    const rows=[...body.children], head=body.closest('table').tHead?.getBoundingClientRect().height||32;
    const size=Math.max(1,Math.min(12,Math.floor((wrap.clientHeight-head-3)/36)));
    pages=Math.max(1,Math.ceil(rows.length/size));page=Math.max(0,Math.min(page,pages-1));
    rows.forEach((row,i)=>{row.hidden=i<page*size||i>=(page+1)*size;});
    label.textContent=(rows.length?'第 '+(page*size+1)+'–'+Math.min(rows.length,(page+1)*size)+' 条':'暂无数据')+' / 共 '+rows.length+' 条';
    previous.disabled=!page;next.disabled=last.disabled=page>=pages-1;
    jump.value=String(page+1);jump.max=String(pages);jump.title='共 '+pages+' 页';
  }
  previous.addEventListener('click',()=>{page--;draw()});
  next.addEventListener('click',()=>{page++;draw()});
  last.addEventListener('click',()=>{page=pages-1;draw()});
  jump.addEventListener('change',()=>{page=(Number(jump.value)||1)-1;draw()});
  new MutationObserver(()=>{page=0;draw()}).observe(body,{childList:true});
  new ResizeObserver(draw).observe(wrap);
  draw();
}

document.addEventListener('DOMContentLoaded',()=>{
  // Expose the already supported four-hour scale in every horizon selector.
  $all('select').filter(select=>[...select.options].some(option=>option.value==='hour_ahead')).forEach(select=>{
    if(![...select.options].some(option=>option.value==='ultrashort'))select.add(new Option('超短期 · 4小时','ultrashort'),1);
  });
  for(const [kind,def] of Object.entries(queryDefinitions)){
    const view=$('.view[data-page="'+def.page+'"]');
    if(!view)continue;
    if(def.horizon){
      const button=document.createElement('button');
      button.type='button';button.className='text-button';button.textContent='返回最新结果';button.dataset.latestResults='';button.hidden=true;
      $(def.hint).after(button);
      button.addEventListener('click',()=>{pinnedRuns.delete(def.page);queryPage(kind)});
    }
    $all('.query-point,.query-range,#resultObject',view).forEach(select=>select.addEventListener('change',()=>queryPage(kind)));
    installQueryAnalytics(kind,def,view);
  }
  $all('.table-wrap tbody').forEach(installTablePager);
});
