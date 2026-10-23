// 全部前端完成，不拆模組，不優化。直接沿用上午 Python 訓練出來的值。

let IRIS = null;
let AI4I = null;

const IRIS_CLASSES = ["setosa", "versicolor", "virginica"];

async function loadModels() {
  try {
    const r1 = await fetch('./iris_model.json');
    IRIS = await r1.json();
  } catch (e) {
    console.error('load iris failed', e);
  }
  try {
    const r2 = await fetch('./ai4i_model.json');
    AI4I = await r2.json();
  } catch (e) {
    console.error('load ai4i failed', e);
  }
}

function softmax(z) {
  let max = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > max) max = z[i];
  let expv = new Array(z.length).fill(0);
  let sum = 0;
  for (let i = 0; i < z.length; i++) { expv[i] = Math.exp(z[i] - max); sum += expv[i]; }
  for (let i = 0; i < z.length; i++) expv[i] = expv[i] / sum;
  return expv;
}

function forwardIris(x) {
  // x is length 4, already scaled? no - we scale here using scaler from model
  let h = new Array(4).fill(0);
  for (let i = 0; i < 4; i++) {
    h[i] = (x[i] - IRIS.scaler.mean[i]) / IRIS.scaler.scale[i];
  }
  // layers
  for (let l = 0; l < IRIS.layers.length - 1; l++) {
    let layer = IRIS.layers[l];
    let w = layer.w; // [in][out]
    let b = layer.b;
    let inDim = h.length;
    let outDim = b.length;
    let nh = new Array(outDim).fill(0);
    for (let j = 0; j < outDim; j++) {
      let s = b[j];
      for (let i = 0; i < inDim; i++) s += h[i] * w[i][j];
      nh[j] = s > 0 ? s : 0; // relu
    }
    h = nh;
  }
  // last layer
  let last = IRIS.layers[IRIS.layers.length - 1];
  let lw = last.w;
  let lb = last.b;
  let lin = h.length;
  let lout = lb.length;
  let z = new Array(lout).fill(0);
  for (let j = 0; j < lout; j++) {
    let s = lb[j];
    for (let i = 0; i < lin; i++) s += h[i] * lw[i][j];
    z[j] = s;
  }
  return softmax(z);
}

function treePredict(tree, x) {
  // tree flat: [feature, threshold, left, right, prob] each node 5 values
  let i = 0;
  while (true) {
    let o = i * 5;
    let left = tree[o + 2];
    if (left === -1) {
      return tree[o + 4];
    }
    let feat = tree[o];
    let thr = tree[o + 1];
    if (x[feat] <= thr) {
      i = left;
    } else {
      i = tree[o + 3];
    }
  }
}

function predictAI4I(form) {
  // form gives: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Type
  let air = parseFloat(form.air.value);
  let proc = parseFloat(form.proc.value);
  let rpm = parseFloat(form.rpm.value);
  let torque = parseFloat(form.torque.value);
  let wear = parseFloat(form.wear.value);
  let typ = form.type.value; // H/L/M

  let raw = [
    air, proc, rpm, torque, wear,
    proc - air,
    torque * 2 * Math.PI * rpm / 60,
    torque * wear,
  ];
  raw.push(typ === 'H' ? 1.0 : 0.0);
  raw.push(typ === 'L' ? 1.0 : 0.0);
  raw.push(typ === 'M' ? 1.0 : 0.0);

  let total = 0.0;
  for (let f = 0; f < AI4I.folds.length; f++) {
    let fold = AI4I.folds[f];
    let x = new Array(11).fill(0);
    // scale first 8
    for (let i = 0; i < 8; i++) {
      x[i] = (raw[i] - AI4I.scaler.mean[i]) / AI4I.scaler.scale[i];
    }
    x[8] = raw[8]; x[9] = raw[9]; x[10] = raw[10];
    let sTree = 0.0;
    let trees = fold.trees;
    for (let t = 0; t < trees.length; t++) {
      sTree += treePredict(trees[t], x);
    }
    let probRaw = sTree / trees.length;
    // sigmoid calibration: 1/(1+exp(a*probRaw + b))
    let zc = fold.a * probRaw + fold.b;
    let probCal = 1.0 / (1.0 + Math.exp(zc));
    total += probCal;
  }
  let prob = total / AI4I.folds.length;
  return prob;
}

function showResult(html) {
  document.getElementById('result').innerHTML = html;
}

window.addEventListener('DOMContentLoaded', async () => {
  await loadModels();
  document.getElementById('status').textContent = 
    'models loaded: iris=' + (IRIS ? 'ok' : 'no') + ', ai4i=' + (AI4I ? 'ok' : 'no');
  
  document.getElementById('irisForm').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!IRIS) { showResult('iris model not loaded'); return; }
    let f = e.target;
    let x = [
      parseFloat(f.sl.value),
      parseFloat(f.sw.value),
      parseFloat(f.pl.value),
      parseFloat(f.pw.value),
    ];
    let p = forwardIris(x);
    let bestIdx = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[bestIdx]) bestIdx = i;
    let html = '<h3>Iris 預測結果</h3>';
    html += '<div>預測類別：<b>' + IRIS_CLASSES[bestIdx] + '</b></div>';
    html += '<div>各類機率：</div><ul>';
    for (let i = 0; i < p.length; i++) {
      html += '<li>' + IRIS_CLASSES[i] + ': ' + (p[i]*100).toFixed(2) + '%</li>';
    }
    html += '</ul>';
    showResult(html);
  });
  
  document.getElementById('ai4iForm').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!AI4I) { showResult('ai4i model not loaded'); return; }
    let prob = predictAI4I(e.target);
    let html = '<h3>AI4I 故障預測結果</h3>';
    html += '<div>故障機率：<b>' + (prob*100).toFixed(4) + '%</b></div>';
    html += '<div>閾值（測試集一次評估）：<b>' + (AI4I.meta.threshold).toFixed(6) + '</b> (' + (AI4I.meta.threshold*100).toFixed(4) + '%)</div>';
    html += '<div>建議：' + (prob >= AI4I.meta.threshold ? '<b style="color:red">預測為故障（>= 閾值）</b>' : '<b style="color:green">預測為正常（< 閾值）</b>') + '</div>';
    showResult(html);
  });
});
