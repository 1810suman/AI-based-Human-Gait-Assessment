<h1>🧠 Freezing of Gait Detection using Machine Learning</h1>

<p>This project focuses on building accurate machine learning models to detect <strong>Freezing of Gait (FoG)</strong>, a key motor symptom associated with the progression of <strong>Parkinson's Disease (PD)</strong>.</p>

<h2>💡 Project Background</h2>

<p>Parkinson's Disease is a progressive neurodegenerative disorder that primarily affects the central nervous system. It leads to movement-related problems such as:</p>
<ul>
    <li>Tremors</li>
    <li>Muscle stiffness</li>
    <li>Balance issues</li>
    <li>Difficulty with walking (gait disorders)</li>
</ul>

<p>One of the most concerning symptoms of Parkinson's is <strong>Freezing of Gait (FoG)</strong>, where individuals suddenly feel as if their feet are "glued" to the ground, making it hard to initiate walking. FoG is often associated with disease progression and severity.</p>

<h3>🎯 Why Detect FoG?</h3>
<p>Detecting FoG episodes accurately can serve as a proxy to assess the presence and progression of Parkinson's Disease, especially in moderate to advanced stages.</p>

<hr>

<h2>📂 Dataset Information</h2>
<p>We utilized the publicly available <strong>Daphnet Freezing of Gait Dataset</strong>, which provides sensor data collected from patients with Parkinson's Disease during walking sessions.</p>
<p><strong>Dataset Link:</strong> <a href="https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait" target="_blank">Daphnet Freezing of Gait Dataset</a></p>

<ul>
    <li>Data collected using wearable sensors (accelerometers) on the lower limbs.</li>
    <li>Labels indicate presence (1) or absence (0) of FoG events.</li>
</ul>

<hr>

<h2>⚙️ Models & Results</h2>
<p>We experimented with three popular machine learning models to classify FoG episodes:</p>

<table border="1" cellpadding="6">
<thead>
    <tr>
        <th>Model</th>
        <th>Optimization</th>
        <th>Test Accuracy</th>
    </tr>
</thead>
<tbody>
    <tr>
        <td>Random Forest</td>
        <td>Default Parameters</td>
        <td>90%</td>
    </tr>
    <tr>
        <td>XGBoost</td>
        <td>Fine-tuned with Optuna</td>
        <td>97%</td>
    </tr>
    <tr>
        <td><strong>CatBoost</strong></td>
        <td><strong>Fine-tuned with Optuna</strong></td>
        <td><strong>98% (Best Model)</strong></td>
    </tr>
</tbody>
</table>

<p>The <strong>CatBoost</strong> model provided the highest accuracy and robustness for FoG detection, making it suitable for further deployment in real-time applications.</p>

<hr>

<h2>🎥 FoG Prediction Video Demo</h2>
<p>Watch the demonstration video showcasing the FoG detection system in action:</p>
<p><a href="https://drive.google.com/file/d/1Nmk-0azCUL6fLEa3HSHZPwjBT0dwXoCx/view?usp=sharing" target="_blank"><strong>▶️ Watch Demo Video</strong></a></p>

<hr>

<h2>🚀 Future Improvements</h2>
<ul>
    <li>Deploy real-time FoG detection using mobile or wearable devices.</li>
    <li>Extend the pipeline for early Parkinson's Disease diagnosis.</li>
    <li>Incorporate signal visualization and severity scoring.</li>
</ul>

<hr>

<h2>📢 Acknowledgements</h2>
<p>Dataset provided by the <a href="https://archive.ics.uci.edu/" target="_blank">UCI Machine Learning Repository</a>.</p>
