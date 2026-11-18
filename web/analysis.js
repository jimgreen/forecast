// Display calculations use the queried records only; missing samples stay missing.
const queryTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
const metricUnits = {wind_speed_mps:'m/s',temperature_c:'°C',solar_radiation_wm2:'W/m²',pv_power_kw:'kW',wind_power_kw:'kW',load_kw:'kW'};
const dayFormatters = new Map();
function queryDay(timestamp, timeZone=queryTimeZone) {
  const date = new Date(timestamp);if(!Number.isFinite(date.getTime()))return '';
  if(!dayFormatters.has(timeZone))dayFormatters.set(timeZone,new Intl.DateTimeFormat('en-CA',{timeZone,year:'numeric',month:'2-digit',day:'2-digit'}));
  const parts=Object.fromEntries(dayFormatters.get(timeZone).formatToParts(date).map(p=>[p.type,p.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}
function fullQueryTime(timestamp) {
  if(!timestamp||!Number.isFinite(Date.parse(timestamp)))return '—';
  return new Date(timestamp).toLocaleString('zh-CN',{timeZone:queryTimeZone,year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
}
function finiteValue(value) {return value==null||value===''?null:Number.isFinite(Number(value))?Number(value):null}
function visibleQueryRows(snapshot) {return !snapshot?.day||snapshot.day==='all'?snapshot?.rows||[]:snapshot.rows.filter(row=>queryDay(row.timestamp)===snapshot.day)}
function summarizeSeries(rows,key) {
  const samples=rows.map(row=>({value:finiteValue(row[key]),time:row.timestamp})).filter(row=>row.value!==null);
  const times=rows.map(row=>Date.parse(row.timestamp)), unique=[...new Set(times.filter(Number.isFinite))].sort((a,b)=>a-b);
  const duplicates=times.filter(Number.isFinite).length-unique.length;
  const irregular=times.some(t=>!Number.isFinite(t)||t%900000!==0);
  const gaps=unique.reduce((sum,t,i)=>sum+(i?Math.max(0,Math.ceil((t-unique[i-1])/900000)-1):0),0);
  const min=samples.reduce((best,row)=>!best||row.value<best.value?row:best,null);
  const max=samples.reduce((best,row)=>!best||row.value>best.value?row:best,null);
  const total=samples.reduce((sum,row)=>sum+row.value,0);
  const integralUnit=key==='solar_radiation_wm2'?'kWh/m²':metricUnits[key]==='kW'?'kWh':null;
  return {count:samples.length,missing:rows.length-samples.length,duplicates,irregular,gaps,min,max,mean:samples.length?total/samples.length:null,
    integral:integralUnit&&samples.length&&!duplicates&&!irregular?total*.25/(key==='solar_radiation_wm2'?1000:1):null,integralUnit};
}
function queryChartScale(rows,specs) {
  const normalized=new Set(specs.map(s=>metricUnits[s.key])).size>1;
  const finite=rows.flatMap(row=>specs.map(s=>finiteValue(row[s.key]))).filter(v=>v!==null);
  let min=finite.length?Math.min(...finite):0,max=finite.length?Math.max(...finite):1;
  if(normalized)return {min:0,max:100,normalized:true,label:'各变量独立归一化 0–100%（读数为原值）'};
  if(min>=0)min=0;
  if(max===min){min-=1;max+=1}
  return {min,max,normalized:false,label:metricUnits[specs[0]?.key]||''};
}
function renderQueryChart(svg,rows,specs) {
  if(!svg)return;
  svg.replaceChildren();svg.setAttribute('viewBox','0 0 1100 360');
  let readout=svg.parentElement.querySelector('.chart-readout');
  if(!readout){readout=document.createElement('p');readout.className='chart-readout';svg.before(readout)}
  readout.textContent=rows.length?'指向曲线或聚焦后按 ← → 查看逐点读数':'暂无数据';
  svg.onpointermove=svg.onclick=svg.onkeydown=null;
  if(!rows.length)return;
  const ns='http://www.w3.org/2000/svg',pad={l:65,r:22,t:30,b:36},width=1100,height=360;
  const add=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,String(v)));if(text!==undefined)e.textContent=text;svg.append(e);return e};
  const times=rows.map(row=>Date.parse(row.timestamp));
  const start=times[0],end=times.at(-1),span=end-start||1;
  const x=i=>pad.l+(width-pad.l-pad.r)*(times[i]-start)/span;
  const scale=queryChartScale(rows,specs);
  add('text',{x:pad.l,y:15,fill:'#9cb5ca','font-size':12},scale.label);
  for(let i=0;i<5;i++){
    const y=pad.t+(height-pad.t-pad.b)*i/4,value=scale.max-(scale.max-scale.min)*i/4;
    add('line',{x1:pad.l,x2:width-pad.r,y1:y,y2:y,stroke:'#294057','stroke-dasharray':'3 5'});
    add('text',{x:pad.l-8,y:y+4,fill:'#9cb5ca','font-size':11,'text-anchor':'end'},num(value,1));
  }
  [...new Set([0,Math.floor((rows.length-1)/3),Math.floor((rows.length-1)*2/3),rows.length-1])].forEach(i=>add('text',{x:x(i),y:height-10,fill:'#9cb5ca','font-size':11,'text-anchor':i===0?'start':i===rows.length-1?'end':'middle'},fmtTime(rows[i].timestamp)));
  specs.forEach(spec=>{
    const values=rows.map(r=>finiteValue(r[spec.key])),finite=values.filter(v=>v!==null);if(!finite.length)return;
    const low=scale.normalized?Math.min(...finite):scale.min,high=scale.normalized?Math.max(...finite):scale.max;
    const y=v=>pad.t+(height-pad.t-pad.b)*(1-(high===low?.5:(v-low)/(high-low)));
    const path=values.map((v,i)=>v===null?'':`${i&&values[i-1]!==null&&times[i]-times[i-1]===900000?'L':'M'} ${x(i)} ${y(v)}`).join(' ');
    add('path',{d:path,fill:'none',stroke:spec.color,'stroke-width':2.2});
    if(rows.length===1)add('circle',{cx:x(0),cy:y(finite[0]),r:4,fill:spec.color});
  });
  const marker=add('line',{x1:x(0),x2:x(0),y1:pad.t,y2:height-pad.b,stroke:'#b7cadd','stroke-dasharray':'3 3'});
  let selected=0;
  const show=i=>{selected=Math.max(0,Math.min(rows.length-1,i));marker.setAttribute('x1',x(selected));marker.setAttribute('x2',x(selected));readout.textContent=fullQueryTime(rows[selected].timestamp)+' · '+specs.map(s=>s.label+' '+num(rows[selected][s.key],2)+' '+metricUnits[s.key]).join(' ｜ ')};
  svg.setAttribute('tabindex','0');svg.setAttribute('role','img');svg.setAttribute('aria-label','预测曲线，方向键查看逐点读数');
  svg.onpointermove=svg.onclick=event=>{const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const cursor=p.matrixTransform(svg.getScreenCTM().inverse()).x;let closest=0;for(let i=1;i<rows.length;i++)if(Math.abs(x(i)-cursor)<Math.abs(x(closest)-cursor))closest=i;show(closest)};
  svg.onkeydown=event=>{if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();show(selected+(event.key==='ArrowLeft'?-1:1))}};
  show(0);
}

function renderQuerySnapshot(kind,snapshot) {
  const def=queryDefinitions[kind], view=$('.view[data-page="'+def.page+'"]');
  if(!def||!view)return;
  const specs=metricSpecs(kind.includes('weather')?'weather':kind,$(def.metric).value);
  renderQueryChart($(def.chart),visibleQueryRows(snapshot),specs);
  renderQueryAnalytics(kind,snapshot);
}

function queryContext(snapshot) {
  if(snapshot.loading)return '查询中…';if(snapshot.error)return '查询失败：'+snapshot.error;
  const rows=visibleQueryRows(snapshot);if(!rows.length)return '当前范围暂无数据';
  const run=snapshot.run;
  const seed=run?.trigger_type==='seed'||rows.some(row=>row.source==='seed');
  const expired=snapshot.horizon&&Date.parse(rows.at(-1).timestamp)<Date.now();
  return [seed?'示例数据':'记录来源：'+[...new Set(rows.map(r=>r.source).filter(Boolean))].join(' / '),expired?'预测区间已结束':'',
    fullQueryTime(rows[0].timestamp)+' — '+fullQueryTime(rows.at(-1).timestamp),queryTimeZone,
    run?'启动 '+fullQueryTime(run.started_at):''].filter(Boolean).join(' · ');
}
function installQueryAnalytics(kind,def,view) {
  const label=document.createElement('label');label.className='day-filter';label.textContent='展示日期';
  const picker=document.createElement('div');picker.className='day-picker';
  const previous=document.createElement('button'),next=document.createElement('button'),select=document.createElement('select');
  previous.type=next.type='button';previous.textContent='‹';next.textContent='›';previous.setAttribute('aria-label','前一天');next.setAttribute('aria-label','后一天');
  select.className='query-day';select.setAttribute('aria-label','展示日期');select.add(new Option('全部日期','all'));
  picker.append(previous,select,next);label.append(picker);$('.query-btn',view).before(label);
  const change=()=>{const snapshot=querySnapshots.get(def.page);if(snapshot){snapshot.day=select.value;renderQuerySnapshot(kind,snapshot)}};
  select.addEventListener('change',change);
  previous.addEventListener('click',()=>{select.selectedIndex=Math.max(0,select.selectedIndex-1);change()});
  next.addEventListener('click',()=>{select.selectedIndex=Math.min(select.options.length-1,select.selectedIndex+1);change()});
  for(const [mode,title] of [['statistics','统计分析'],['daily','逐日汇总']]){
    const tabs=$('.page-tabs',view),content=$('.page-tab-content',view),index=tabs.children.length;
    const tab=document.createElement('button');tab.type='button';tab.className='page-tab';tab.textContent=title;tab.addEventListener('click',()=>activatePageTab(view,index));tabs.append(tab);
    const panel=document.createElement('div');panel.className='sub-panel';panel.dataset.tabPanel=String(index);
    panel.innerHTML=`<article class="panel analytics-panel"><div class="panel-header"><div><h2>${title}</h2><p class="query-context"></p></div><button class="secondary-button summary-export" disabled>导出统计</button></div><div class="table-wrap"><table><thead><tr>${(mode==='daily'?['日期','变量','最小值','最大值','平均值','有效/缺值','累计量']:['变量 / 单位','最小值 / 时刻','最大值 / 时刻','平均值','有效/缺值','累计量']).map(t=>'<th>'+t+'</th>').join('')}</tr></thead><tbody data-analysis="${mode}"></tbody></table></div><p class="analysis-note">按有效15分钟样本累计，缺测不补零；重复时刻或不规则时间轴不累计。</p></article>`;
    content.append(panel);
    $('.summary-export',panel).addEventListener('click',()=>exportQuerySummary(kind,mode));
  }
}
function renderQueryAnalytics(kind,snapshot) {
  const def=queryDefinitions[kind],view=$('.view[data-page="'+def.page+'"]'),select=$('.query-day',view);
  if(!select)return;
  const days=[...new Set(snapshot.rows.map(row=>queryDay(row.timestamp)).filter(Boolean))];
  const desired=['all',...days];
  if([...select.options].map(o=>o.value).join('|')!==desired.join('|'))select.replaceChildren(...desired.map(day=>new Option(day==='all'?'全部日期':day,day)));
  if(!desired.includes(snapshot.day))snapshot.day='all';select.value=snapshot.day;
  select.disabled=snapshot.loading||!snapshot.rows.length;
  const buttons=$all('.day-picker button',view);buttons[0].disabled=select.disabled||select.selectedIndex===0;buttons[1].disabled=select.disabled||select.selectedIndex===select.options.length-1;
  const sourceRows=visibleQueryRows(snapshot),specs=metricSpecs(kind.includes('weather')?'weather':kind,$(def.metric).value);
  const rows=kind==='energy'?sourceRows.filter(row=>typeof authoritativeEnergyRow==='function'?authoritativeEnergyRow(row):true):sourceRows;
  const excluded=sourceRows.length-rows.length,unknownEnergy=kind==='energy'&&sourceRows.length&&!rows.length;
  $all('.query-context',view).forEach(el=>{el.textContent=queryContext(snapshot);el.title=el.textContent});
  $all('.summary-export',view).forEach(el=>el.disabled=snapshot.loading||!!snapshot.error||!rows.length);
  const stats=specs.map(spec=>({spec,stat:summarizeSeries(rows,spec.key)}));
  $('[data-analysis="statistics"]',view).innerHTML=unknownEnergy?specs.map(spec=>`<tr><td>${spec.label} / ${metricUnits[spec.key]}</td><td colspan="5">未知（测试/样例来源）</td></tr>`).join(''):rows.length?stats.map(({spec,stat})=>{
    const range=entry=>entry?num(entry.value,2)+'<small>'+escapeHtml(fullQueryTime(entry.time))+'</small>':'—';
    return `<tr><td>${spec.label} / ${metricUnits[spec.key]}</td><td>${range(stat.min)}</td><td>${range(stat.max)}</td><td>${num(stat.mean,2)}</td><td>${stat.count} / ${stat.missing}</td><td>${stat.integral===null?'—':num(stat.integral,2)+' '+stat.integralUnit}</td></tr>`;
  }).join(''):'';
  const daily=unknownEnergy?unknownDailySummary(sourceRows,specs):buildDailySummary(rows,specs);
  $('[data-analysis="daily"]',view).innerHTML=daily.map(row=>`<tr><td>${row.date}</td><td>${row.metric} / ${row.unit}</td><td>${num(row.min,2)}</td><td>${num(row.max,2)}</td><td>${num(row.mean,2)}</td><td>${row.count} / ${row.missing}</td><td>${row.integral===null?'—':num(row.integral,2)+' '+row.integral_unit}</td></tr>`).join('');
  const health=stats[0]?.stat;
  $all('.analysis-note',view).forEach(el=>el.textContent=unknownEnergy?'当前查询仅含测试/样例来源；最值、均值、累计量和弃电信息未知。':('有效15分钟样本累计，缺测不补零。'+(excluded?`已排除 ${excluded} 条测试/样例记录。`:'')+(health?`时间轴缺口 ${health.gaps} 点；重复时刻 ${health.duplicates} 个。`:'')+(health?.irregular?'时间轴不规则，累计量停用。':'')));
}
function buildDailySummary(rows,specs) {
  const groups=new Map();for(const row of rows){const day=queryDay(row.timestamp);if(!groups.has(day))groups.set(day,[]);groups.get(day).push(row)}
  return [...groups].flatMap(([date,items])=>specs.map(spec=>{const s=summarizeSeries(items,spec.key);return {date,metric:spec.label,unit:metricUnits[spec.key],min:s.min?.value??null,max:s.max?.value??null,mean:s.mean,count:s.count,missing:s.missing,integral:s.integral,integral_unit:s.integralUnit||'',gaps:s.gaps,duplicates:s.duplicates}}));
}
function unknownDailySummary(sourceRows,specs) {
  const days=[...new Set(sourceRows.map(row=>queryDay(row.timestamp)).filter(Boolean))];
  return days.flatMap(date=>specs.map(spec=>({date,metric:spec.label,unit:metricUnits[spec.key],min:null,max:null,mean:null,count:0,missing:sourceRows.filter(row=>queryDay(row.timestamp)===date).length,integral:null,integral_unit:''})));
}
function exportQuerySummary(kind,mode) {
  const def=queryDefinitions[kind],snapshot=querySnapshots.get(def.page);if(!snapshot||snapshot.loading||snapshot.error)return;
  const sourceRows=visibleQueryRows(snapshot),specs=metricSpecs(kind.includes('weather')?'weather':kind,$(def.metric).value);
  const rows=kind==='energy'?sourceRows.filter(row=>typeof authoritativeEnergyRow==='function'?authoritativeEnergyRow(row):true):sourceRows;
  if(kind==='energy'&&sourceRows.length&&!rows.length)return;
  const summary=mode==='daily'?buildDailySummary(rows,specs):specs.map(spec=>{const s=summarizeSeries(rows,spec.key);return {metric:spec.label,unit:metricUnits[spec.key],min:s.min?.value??null,min_time:s.min?.time||'',max:s.max?.value??null,max_time:s.max?.time||'',mean:s.mean,count:s.count,missing:s.missing,integral:s.integral,integral_unit:s.integralUnit||'',gaps:s.gaps,duplicates:s.duplicates}});
  const exported=summary.map(row=>({...row,point_id:snapshot.point,object_id:snapshot.objectId||'',run_id:snapshot.runId||'',horizon:snapshot.horizon||'',time_zone:queryTimeZone}));
  if(exported.length)downloadCSV(exported,Object.keys(exported[0]),'forecast-'+mode+'-'+(snapshot.runId||snapshot.point)+'-'+(snapshot.day||'all'));
}
