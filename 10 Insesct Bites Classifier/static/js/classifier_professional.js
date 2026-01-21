// Classifier JavaScript with Professional Medical Analysis

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const errorMessage = document.getElementById('errorMessage');
const results = document.getElementById('results');
let selectedFile = null;

// Health check on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        if (!response.ok) {
            console.warn('Health check failed:', response.status);
        } else {
            const data = await response.json();
            console.log('Server health:', data);
        }
    } catch (err) {
        console.warn('Could not reach server:', err.message);
    }
});
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    selectedFile = file;
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPG, PNG, GIF, BMP)');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showError('File size must be less than 16MB');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        analyzeBtn.style.display = 'block';
        errorMessage.style.display = 'none';
        results.style.display = 'none';
    };    
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    loading.style.display = 'block';
    analyzeBtn.disabled = true;
    errorMessage.style.display = 'none';
    results.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        console.log('Sending prediction request...');
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
            const errorMsg = errorData.error || `Server error: ${response.status}`;
            console.error('Server error:', errorMsg);
            showError(errorMsg);
            return;
        }

        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.success) {
            displayResults(data);
        } else {
            const msg = data.error || 'Prediction failed for unknown reason';
            console.error('Prediction failed:', msg);
            showError(msg);
        }
    } catch (err) {
        console.error('Fetch error:', err);
        showError('Network error: ' + err.message + '. Check console for details.');
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function displayResults(data) {
    document.getElementById('predictedClass').textContent = 
        data.predicted_class.replace('_', ' ').toUpperCase();
    document.getElementById('confidence').textContent = 
        `Confidence: ${data.confidence_percent}`;

    const topPredEl = document.getElementById('topPredictions');
    topPredEl.innerHTML = '';
    
    data.top_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="class-name">#${index + 1} - ${pred.class.replace('_', ' ')}</span>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
            </div>
            <span class="confidence-value">${pred.confidence_percent}</span>
        `;
        topPredEl.appendChild(item);
    });
    
    generateMedicalAnalysis(data.top_predictions);

    const firstAid = data.first_aid;
    if (firstAid) {
        document.getElementById('symptoms').textContent = firstAid.symptoms || 'N/A';
        
        const stepsEl = document.getElementById('firstAidSteps');
        stepsEl.innerHTML = '';
        if (firstAid.first_aid) {
            firstAid.first_aid.forEach(step => {
                const li = document.createElement('li');
                li.textContent = step;
                stepsEl.appendChild(li);
            });
        }
        
        document.getElementById('whenToSeekHelp').textContent = firstAid.when_to_seek_help || 'N/A';
        
        const severityEl = document.getElementById('severity');
        severityEl.textContent = firstAid.severity || 'N/A';
        severityEl.className = 'severity-badge';
        if (firstAid.severity && firstAid.severity.toLowerCase().includes('low')) {
            severityEl.classList.add('severity-low');
        } else if (firstAid.severity && firstAid.severity.toLowerCase().includes('moderate')) {
            severityEl.classList.add('severity-moderate');
        } else {
            severityEl.classList.add('severity-high');
        }
        
        document.getElementById('prevention').textContent = firstAid.prevention || 'N/A';
    }

    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth' });
}

function generateMedicalAnalysis(predictions) {
    const analysisEl = document.getElementById('predictionAnalysis');
    if (!predictions || predictions.length === 0) {
        analysisEl.innerHTML = '';
        return;
    }

    const normalizeKey = name => String(name || '').toLowerCase().replace(/\s+/g, '_');
    const getConfidence = p => {
        if (!p) return 0;
        return (typeof p.confidence === 'number') ? Math.round(p.confidence * 100) : 
               (p.confidence_percent ? parseInt(String(p.confidence_percent).replace('%','')) : 0);
    };

    const top = predictions[0] || {};
    const second = predictions[1] || {};
    const third = predictions[2] || {};

    const conf0 = getConfidence(top);
    const conf1 = getConfidence(second);
    const conf2 = getConfidence(third);
    const delta01 = conf0 - conf1;

    const topKey = normalizeKey(top.class);
    const topMedical = medicalDatabase[topKey] || {};

    let html = `
        <div style="margin-bottom: 15px;">
            <strong>Primary Diagnosis Assessment: ${top.class.replace('_', ' ').toUpperCase()}</strong> (${conf0}% confidence)
            <ul style="margin: 8px 0 12px 20px; color: var(--gh-text-primary);">
                <li><strong>Morphology:</strong> ${topMedical.morphology || 'Pattern recognition analysis'}</li>
                <li><strong>Typical Distribution:</strong> ${topMedical.distribution || 'Variable location'}</li>
                <li><strong>Expected Presentation:</strong> ${topMedical.symptoms || 'Clinical manifestations'}</li>
                <li><strong>Symptom Timeline:</strong> ${topMedical.timeline || 'Time course pending clinical observation'}</li>
                <li><strong>Severity Classification:</strong> ${topMedical.severity || 'Requires assessment'}</li>
                <li><strong>Notable Complications:</strong> ${topMedical.complications || 'Low incidence of complications'}</li>
            </ul>
        </div>

        <div style="margin-bottom: 15px; padding: 10px; background: rgba(237, 125, 49, 0.1); border-left: 3px solid #ED7D31; border-radius: 4px;">
            <strong>Differential Diagnoses</strong>
            <ul style="margin: 8px 0 0 20px; color: var(--gh-text-primary);">
                <li><strong>#2: ${second.class ? second.class.replace('_', ' ') : 'N/A'}</strong> (${conf1}%) — Consider if atypical features present; confidence delta ${delta01}%</li>
                <li><strong>#3: ${third.class ? third.class.replace('_', ' ') : 'N/A'}</strong> (${conf2}%) — Lower probability; detailed clinical correlation recommended</li>
            </ul>
        </div>

        <div style="padding: 10px; background: rgba(70, 173, 70, 0.1); border-left: 3px solid #70AD47; border-radius: 4px;">
            <strong>Clinical Recommendation</strong>
            <p style="margin: 8px 0; color: var(--gh-text-primary);">
                This AI-assisted analysis provides preliminary pattern recognition support. Final diagnosis and treatment decisions must be made by qualified healthcare professionals following direct clinical examination. Consider dermoscopy, full patient history (exposure, timeline, associated symptoms), and systemic evaluation where clinically indicated.
            </p>
        </div>
    `;
    
    analysisEl.innerHTML = html;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Tab switching
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tabName = btn.getAttribute('data-tab');
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(tc => tc.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(tabName).classList.add('active');
    });
});

// Camera functionality
let mediaStream = null;
const capturePhotoBtn = document.getElementById('capturePhotoBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const startCameraBtn = document.getElementById('startCameraBtn');
const canvas = document.getElementById('canvasPreview');
const video = document.getElementById('videoPreview');

if (startCameraBtn && stopCameraBtn && capturePhotoBtn && canvas && video) {
    // Ensure inline playback for mobile and muted to allow autoplay in some browsers
    video.setAttribute('playsinline', 'true');
    video.muted = true;
    // Keep video hidden; we only need the frame for a still capture (overlay handles controls)
    video.style.display = 'block';
    video.style.width = '100%';
    video.style.maxHeight = '480px';
    video.style.objectFit = 'cover';

    const stopStream = () => {
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        video.srcObject = null;
        startCameraBtn.disabled = false;
        stopCameraBtn.disabled = true;
        capturePhotoBtn.disabled = true;
    };

    startCameraBtn.addEventListener('click', async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Camera not supported in this browser/device.');
            return;
        }
        try {
            // Stop any existing stream before starting a new one
            stopStream();
            mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });
            video.srcObject = mediaStream;
            await video.play();
            // Wait for metadata so dimensions are available for capture
            if (video.readyState < 2) {
                await new Promise(resolve => {
                    video.onloadedmetadata = () => resolve();
                });
            }
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            capturePhotoBtn.disabled = false;
            canvas.style.display = 'none';
            video.style.display = 'none';
        } catch (err) {
            showError('Camera access denied or not available: ' + err.message);
        }
    });

    stopCameraBtn.addEventListener('click', () => {
        stopStream();
    });

    capturePhotoBtn.addEventListener('click', () => {
        const ctx = canvas.getContext('2d');
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const size = Math.min(vw, vh);
        const sx = (vw - size) / 2;
        const sy = (vh - size) / 2;
        canvas.width = size;
        canvas.height = size;
        ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);
        
        canvas.toBlob(blob => {
            selectedFile = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
            canvas.style.display = 'block';
            analyzeBtn.style.display = 'block';
            errorMessage.style.display = 'none';
            results.style.display = 'none';
            // Stop stream after taking a still image so it behaves like a photo capture, not video
            stopStream();
        }, 'image/jpeg', 0.95);
    });
}
