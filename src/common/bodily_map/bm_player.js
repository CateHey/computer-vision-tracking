(function(){
var D=(typeof BM_DATA!=='undefined')?BM_DATA:null;
var TCOL=['#2563EB','#E0489B','#0FB6C4','#E69F00'],RECENT=1.5;
function pad(n){return (n<10?'0':'')+n}
function fmt(t,dec){t=Math.max(0,t);var m=Math.floor(t/60),s=t-m*60;
  return pad(m)+':'+(dec?(s<10?'0':'')+s.toFixed(1):pad(Math.floor(s)))}
var MAGMA=[[0,[0,0,4]],[.25,[81,18,124]],[.5,[183,55,121]],[.75,[252,137,97]],[1,[252,253,191]]];
function magma(v){for(var i=1;i<MAGMA.length;i++){if(v<=MAGMA[i][0]){var a=MAGMA[i-1],b=MAGMA[i],f=(v-a[0])/(b[0]-a[0]);
  return [a[1][0]+(b[1][0]-a[1][0])*f,a[1][1]+(b[1][1]-a[1][1])*f,a[1][2]+(b[1][2]-a[1][2])*f]}}return MAGMA[4][1]}
function el(tag,cls,html){var e=document.createElement(tag);if(cls)e.className=cls;if(html!=null)e.innerHTML=html;return e}

function BMPlayer(root){
  var R=D.records,TP=D.template,n=R.time.length;
  var VIEWS=['all','passive','active','mutual'];
  var s={focal:-1,cat:'all',view:(VIEWS.indexOf(root.dataset.view)>=0?root.dataset.view:'all'),color:'mouse',heat:false,T:0,playing:false,speed:1,last:0,sel:[],byType:[[],[],[],[]],count:0};
  var f0=root.dataset.focal;if(f0){var k=D.mice.indexOf(f0);if(k>=0)s.focal=k}
  var rate=Math.max(1,D.t_end/45);
  root.innerHTML='';
  var bar=el('div','bmp-bar');
  var bPlay=el('button','btn play','Play');bPlay.type='button';bPlay.setAttribute('aria-label','Play or pause the animation');
  var bReset=el('button','btn','Reset');bReset.type='button';bReset.setAttribute('aria-label','Reset time to zero');
  var tr=el('span','bmp-time','00:00');
  function sel(label,opts,aria){var l=el('label','',label+' ');var x=document.createElement('select');x.setAttribute('aria-label',aria);
    opts.forEach(function(o){var op=document.createElement('option');op.value=o[0];op.textContent=o[1];x.appendChild(op)});l.appendChild(x);return [l,x]}
  var mo=[['-1','All mice']].concat(D.mice.map(function(m,i){return [String(i),m]}));
  var sSpeed=sel('Speed',[['1','1x'],['2','2x'],['4','4x']],'Playback speed');
  var sMouse=sel('Mouse',mo,'Which mouse to show');
  var sView=sel('View',[['all','Touched — all'],['passive','Passive (directional)'],['active','Active / Touching (directional)'],['mutual','Mutual (N2N + SBS)']],'Which contacts to show for this mouse: all touches on its body, directional passive, directional active, or mutual');
  var sCat=sel('Category',[['all','All'],['dir','Directional'],['mut','Mutual']],'Category filter: all, directional (N2AG, N2B) or mutual (N2N, SBS)');
  var sCol=sel('Colour by',[['mouse','Mouse'],['type','Contact type']],'Colour dots by');
  var lh=el('label','','<input type="checkbox"> Density');var cHeat=lh.firstChild;cHeat.setAttribute('aria-label','Show density heatmap');
  [bPlay,bReset,tr,sSpeed[0],sMouse[0],sView[0],sCat[0],sCol[0],lh].forEach(function(x){bar.appendChild(x)});
  var slider=document.createElement('input');slider.type='range';slider.className='bmp-slider';slider.min=0;slider.max=D.t_end;slider.step=0.1;slider.value=0;
  slider.setAttribute('aria-label','Session time');
  var plot=el('div','bmp-plot'),cv=document.createElement('canvas');cv.setAttribute('role','img');
  cv.setAttribute('aria-label','Template mouse with contact points accumulating over time');plot.appendChild(cv);
  var tlw=el('div','bmp-plot bmp-tl'),tc=document.createElement('canvas');tc.setAttribute('role','img');
  tc.setAttribute('aria-label','Timeline of contacts by type with time cursor');tlw.appendChild(tc);
  var tip=el('div','bmp-tip');tlw.appendChild(tip);
  var foot=el('div','bmp-foot'),leg=el('div','legend'),note=el('span','note');foot.appendChild(leg);foot.appendChild(note);
  var mnote=el('p','note','Hollow rings = mutual contacts: N2N (nose-to-nose) and SBS (side-by-side) have no active/passive direction, both mice investigate each other (as in the summary figure). Filled dots = directional (N2AG, N2B). A mutual contact is shown once on each mouse.');
  [bar,slider,plot,tlw,foot,mnote].forEach(function(x){root.appendChild(x)});
  sMouse[1].value=String(s.focal);sView[1].value=s.view;
  var ctx=cv.getContext('2d'),tx=tc.getContext('2d');
  var ratio=TP.aspect*(TP.smax-TP.smin)/(TP.tmax-TP.tmin);
  var W=0,H=0,TW=0,TH=0,dpr=1,heatCv=document.createElement('canvas'),GW=120,GH=54;heatCv.width=GW;heatCv.height=GH;
  var heatN=-2;

  function isMut(i){var t=R.type[i];return t===0||t===3}
  function recalc(){
    s.sel=[];s.byType=[[],[],[],[]];
    for(var i=0;i<n;i++){
      var mu=isMut(i),ok;
      if(s.focal<0)ok=s.cat==='all'||(s.cat==='mut')===mu;
      else if(s.view==='all')ok=R.focal[i]===s.focal;
      else if(s.view==='passive')ok=!mu&&R.focal[i]===s.focal;
      else if(s.view==='active')ok=!mu&&R.other[i]===s.focal;
      else ok=mu&&R.focal[i]===s.focal;
      if(ok){s.sel.push(i);s.byType[R.type[i]].push(i)}}
    heatN=-2;
    sView[1].disabled=s.focal<0;sCat[1].disabled=s.focal>=0;
  }
  function upper(list,t){var lo=0,hi=list.length;while(lo<hi){var m=(lo+hi)>>1;if(R.time[list[m]]<=t)lo=m+1;else hi=m}return lo}
  function col(i){
    if(s.color==='type')return TCOL[R.type[i]];
    return D.colors[(s.focal>=0&&s.view==='active')?R.focal[i]:R.other[i]]}
  function legend(){
    var h='';
    if(s.color==='type')D.types.forEach(function(t,i){h+='<span class="lg"><span class="dot" style="background:'+TCOL[i]+'"></span>'+t+'</span>'});
    else D.mice.forEach(function(m,i){h+='<span class="lg"><span class="dot" style="background:'+D.colors[i]+'"></span>'+m+'</span>'});
    leg.innerHTML=h}
  // (t,s) -> pixels: same extents and aspect as template.py (x spans tmin..tmax, y spans smax..smin, s>0 up).
  function px(t,sv){return [(t-TP.tmin)/(TP.tmax-TP.tmin)*W,(TP.smax-sv)/(TP.smax-TP.smin)*H]}
  function path(c,pts){c.beginPath();for(var i=0;i<pts.length;i++){var p=px(pts[i][0],pts[i][1]);if(i)c.lineTo(p[0],p[1]);else c.moveTo(p[0],p[1])}c.closePath()}
  function drawTemplate(){
    var c=ctx;c.save();c.lineJoin='round';
    c.save();c.translate(W*.004,H*.007);path(c,TP.body);c.fillStyle='rgba(0,0,0,.07)';c.fill();c.restore();
    TP.ears.forEach(function(e){path(c,e);c.fillStyle='#ecc9c0';c.fill();c.strokeStyle='#4d4d4d';c.lineWidth=1;c.stroke()});
    path(c,TP.tail);c.fillStyle='#e9dfd0';c.fill();c.strokeStyle='#4d4d4d';c.lineWidth=1;c.stroke();
    path(c,TP.body);c.fillStyle='#f5f1e8';c.fill();
    c.setLineDash([3,3]);c.strokeStyle='#8a8a8a';c.lineWidth=.6;
    Object.keys(TP.regions).forEach(function(k){TP.regions[k].forEach(function(v){path(c,v);c.stroke()})});
    c.setLineDash([]);path(c,TP.body);c.strokeStyle='#4d4d4d';c.lineWidth=1.4;c.stroke();
    c.fillStyle='#333';TP.eyes.forEach(function(e){var p=px(e[0],e[1]);c.beginPath();c.arc(p[0],p[1],Math.max(1.5,W/450),0,6.283);c.fill()});
    c.restore()}
  function drawHeat(cnt){
    if(heatN!==cnt){
      var g=new Float32Array(GW*GH),KR=3,mx=0;
      for(var q=0;q<cnt;q++){var i=s.sel[q],gx=(R.tt[i]-TP.tmin)/(TP.tmax-TP.tmin)*GW,gy=(TP.smax-R.ss[i])/(TP.smax-TP.smin)*GH,w=R.dur[i]||.03;
        for(var y=Math.max(0,Math.floor(gy)-KR);y<=Math.min(GH-1,Math.floor(gy)+KR);y++)for(var x=Math.max(0,Math.floor(gx)-KR);x<=Math.min(GW-1,Math.floor(gx)+KR);x++){
          var d2=(x+.5-gx)*(x+.5-gx)+(y+.5-gy)*(y+.5-gy);g[y*GW+x]+=w*Math.exp(-d2/2)}}
      for(var k=0;k<g.length;k++)if(g[k]>mx)mx=g[k];
      var hc=heatCv.getContext('2d'),im=hc.createImageData(GW,GH);
      for(var k2=0;k2<g.length;k2++){var v=mx>0?Math.sqrt(g[k2]/mx):0,cc=magma(v);
        im.data[k2*4]=cc[0];im.data[k2*4+1]=cc[1];im.data[k2*4+2]=cc[2];im.data[k2*4+3]=v<.04?0:Math.min(255,60+v*190)}
      hc.putImageData(im,0,0);heatN=cnt}
    ctx.save();ctx.imageSmoothingEnabled=true;ctx.globalAlpha=.9;ctx.drawImage(heatCv,0,0,W,H);ctx.restore()}
  function drawMap(){
    if(!W)return;
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
    drawTemplate();
    var cnt=upper(s.sel,s.T);s.count=cnt;
    if(s.heat)drawHeat(cnt);
    var r0=Math.max(2.2,Math.min(5,W/160)),c=ctx,Tn=s.T;
    c.lineWidth=.8;
    for(var q=0;q<cnt;q++){var i=s.sel[q],age=Tn-R.time[i];if(age<=RECENT)continue;
      var p=px(R.tt[i],R.ss[i]);c.globalAlpha=.5;c.fillStyle=col(i);c.strokeStyle='rgba(20,20,40,.55)';
      c.beginPath();c.arc(p[0],p[1],r0*.72,0,6.283);
      if(isMut(i)){c.globalAlpha=.8;c.fillStyle='#fff';c.fill();c.strokeStyle=col(i);c.lineWidth=1.6;c.stroke();c.lineWidth=.8}else{c.fill();c.stroke()}}
    for(var q2=Math.max(0,cnt-4000);q2<cnt;q2++){var j=s.sel[q2],a=Tn-R.time[j];if(a>RECENT)continue;
      var f=1-a/RECENT,pp=px(R.tt[j],R.ss[j]),cl=col(j);
      c.globalAlpha=.28*f;c.fillStyle=cl;c.beginPath();c.arc(pp[0],pp[1],r0*(2.2+1.2*f),0,6.283);c.fill();
      c.globalAlpha=1;c.lineWidth=1.2;c.strokeStyle='#14152A';c.fillStyle=cl;
      c.beginPath();c.arc(pp[0],pp[1],r0*(1+.6*f),0,6.283);
      if(isMut(j)){c.fillStyle='#fff';c.fill();c.stroke();c.beginPath();c.arc(pp[0],pp[1],r0*(1+.6*f)-1.2,0,6.283);c.strokeStyle=cl;c.lineWidth=2;c.stroke()}else{c.fill();c.stroke()}}
    c.globalAlpha=1}
  var GUT=46,ROWH=0,TOP=6;
  function tlx(t){return GUT+t/D.t_end*(TW-GUT-8)}
  function drawTL(){
    if(!TW)return;
    var c=tx;c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,TW,TH);
    ROWH=(TH-TOP-22)/4;
    c.font='11px system-ui,sans-serif';c.textBaseline='middle';
    for(var r=0;r<4;r++){var y=TOP+r*ROWH;
      c.fillStyle=r%2?'#f6f6fb':'#eeeef8';c.fillRect(GUT,y,TW-GUT-8,ROWH);
      c.fillStyle='#40456A';c.textAlign='left';c.fillText(D.types[r],4,y+ROWH/2);
      var L=s.byType[r];
      for(var q=0;q<L.length;q++){var i=L[q],x=tlx(R.time[i]);
        c.globalAlpha=R.time[i]<=s.T?.95:.22;c.fillStyle=col(i);c.fillRect(x-.6,y+3,Math.max(1.6,R.dur[i]/D.t_end*(TW-GUT-8)),ROWH-6)}
      c.globalAlpha=1}
    c.textBaseline='top';c.textAlign='center';c.fillStyle='#686F96';c.strokeStyle='#C7CBE4';c.lineWidth=1;
    var span=D.t_end,steps=[1,2,5,10,15,30,60,120,300,600],st=steps[steps.length-1];
    for(var k=0;k<steps.length;k++){if(span/steps[k]<=Math.max(3,(TW-GUT)/70)){st=steps[k];break}}
    for(var t=0;t<=span+1e-6;t+=st){var xx=Math.round(tlx(t))+.5;c.beginPath();c.moveTo(xx,TOP+4*ROWH);c.lineTo(xx,TOP+4*ROWH+4);c.stroke();c.fillText(fmt(t),xx,TOP+4*ROWH+6)}
    var cx=tlx(s.T);c.strokeStyle='#2563EB';c.fillStyle='#2563EB';c.lineWidth=2;
    c.beginPath();c.moveTo(cx,TOP-2);c.lineTo(cx,TOP+4*ROWH);c.stroke();
    c.beginPath();c.moveTo(cx-5,0);c.lineTo(cx+5,0);c.lineTo(cx,6);c.closePath();c.fill()}
  function update(){
    slider.value=s.T;tr.textContent=fmt(s.T)+' / '+fmt(D.t_end);
    slider.setAttribute('aria-valuetext',fmt(s.T));
    drawMap();drawTL();
    var who=s.focal<0?'all mice':(D.mice[s.focal]+': '+sView[1].options[sView[1].selectedIndex].textContent);
    var dir=0,mu=0,dirT=0,muT=0;
    for(var q=0;q<s.sel.length;q++){var m=isMut(s.sel[q]);if(m)muT++;else dirT++;if(q<s.count){if(m)mu++;else dir++}}
    var ev=s.focal<0?(dir+Math.round(mu/2)):(dir+mu),evT=s.focal<0?(dirT+Math.round(muT/2)):(dirT+muT);
    note.textContent=ev+' of '+evT+' touch events shown ('+who+'; mutual contacts counted once; last '+RECENT+' s highlighted)'+
      (s.focal>=0&&s.view==='active'&&s.color==='mouse'?'. Colour = touched mouse.':'')}
  function resize(){
    var w=Math.floor(plot.clientWidth-14);if(w<40)return;
    dpr=window.devicePixelRatio||1;W=w;H=Math.round(w*ratio);
    cv.style.height=H+'px';cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);
    TW=Math.floor(tlw.clientWidth-14);TH=TW<420?150:132;tc.style.height=TH+'px';tc.width=Math.round(TW*dpr);tc.height=Math.round(TH*dpr);
    update()}
  function setT(t){s.T=Math.max(0,Math.min(D.t_end,t));update()}
  function pause(){s.playing=false;bPlay.textContent='Play'}
  function tick(ts){
    if(!s.playing)return;
    var dt=Math.min(.1,(ts-s.last)/1000);s.last=ts;s.T+=dt*rate*s.speed;
    if(s.T>=D.t_end){s.T=D.t_end;pause()}
    update();if(s.playing)requestAnimationFrame(tick)}
  function play(){
    if(s.playing){pause();return}
    if(s.T>=D.t_end)s.T=0;
    s.playing=true;bPlay.textContent='Pause';s.last=performance.now();requestAnimationFrame(tick)}
  bPlay.addEventListener('click',play);
  bReset.addEventListener('click',function(){pause();setT(0)});
  slider.addEventListener('input',function(){setT(parseFloat(slider.value))});
  sSpeed[1].addEventListener('change',function(){s.speed=parseFloat(sSpeed[1].value)});
  sMouse[1].addEventListener('change',function(){s.focal=parseInt(sMouse[1].value,10);recalc();update()});
  sView[1].addEventListener('change',function(){s.view=sView[1].value;recalc();update()});
  sCat[1].addEventListener('change',function(){s.cat=sCat[1].value;recalc();update()});
  sCol[1].addEventListener('change',function(){s.color=sCol[1].value;legend();update()});
  cHeat.addEventListener('change',function(){s.heat=cHeat.checked;update()});
  function tlT(e){var b=tc.getBoundingClientRect();return [(e.clientX-b.left-GUT)/(b.width-GUT-8)*D.t_end,e.clientX-b.left,e.clientY-b.top]}
  var drag=false;
  tc.addEventListener('pointerdown',function(e){drag=true;try{tc.setPointerCapture(e.pointerId)}catch(x){}setT(tlT(e)[0])});
  tc.addEventListener('pointerup',function(){drag=false});
  tc.addEventListener('pointerleave',function(){tip.style.display='none'});
  tc.addEventListener('pointermove',function(e){
    var q=tlT(e);if(drag){setT(q[0]);return}
    var row=Math.floor((q[2]-TOP)/ROWH);if(row<0||row>3||!ROWH){tip.style.display='none';return}
    var L=s.byType[row],k=upper(L,q[0]),best=-1,bd=1e9;
    for(var z=Math.max(0,k-3);z<Math.min(L.length,k+3);z++){var d=Math.abs(tlx(R.time[L[z]])-q[1]);if(d<bd){bd=d;best=L[z]}}
    if(best<0||bd>7){tip.style.display='none';return}
    tip.textContent=fmt(R.time[best],1)+'  '+D.types[R.type[best]]+'  '+D.mice[R.other[best]]+' → '+D.mice[R.focal[best]]+
      (R.region[best]>=0?'  ('+D.regions[R.region[best]].replace('_',' ')+')':'');
    tip.style.display='block';var pr=tlw.getBoundingClientRect(),tb=tc.getBoundingClientRect();
    tip.style.left=Math.max(0,Math.min(pr.width-tip.offsetWidth,q[1]+(tb.left-pr.left)-tip.offsetWidth/2))+'px';
    tip.style.top=(q[2]+(tb.top-pr.top)-30)+'px'});
  root.addEventListener('keydown',function(e){
    var tg=e.target.tagName;
    if(e.key===' '||e.key==='Spacebar'){if(tg==='BUTTON'||tg==='SELECT'||tg==='INPUT')return;e.preventDefault();play()}
    else if((e.key==='ArrowRight'||e.key==='ArrowLeft')&&tg!=='INPUT'&&tg!=='SELECT'){
      e.preventDefault();var stp=Math.max(1,D.t_end/100);setT(s.T+(e.key==='ArrowRight'?stp:-stp))}});
  recalc();legend();
  if(window.ResizeObserver)new ResizeObserver(resize).observe(root);else window.addEventListener('resize',resize);
  root.classList.add('ready');resize();
  this.setT=setT;this.play=play;this.state=s;
}
window.BMPlayer=BMPlayer;
window.bmInit=function(id){
  var p=document.getElementById(id);if(!p)return;
  [].slice.call(p.querySelectorAll('.bmp:not(.ready)')).forEach(function(r){
    if(!D||!D.records||!D.records.time.length||!(D.t_end>0)){r.innerHTML='<p class="empty">No contacts to animate.</p>';r.classList.add('ready');return}
    try{r._bm=new BMPlayer(r)}catch(e){r.innerHTML='<p class="empty">Interactive map unavailable in this browser.</p>';r.classList.add('ready')}})};
})();
