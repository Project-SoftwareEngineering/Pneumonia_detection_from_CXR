const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const resultsCard = document.getElementById('results-card');

// NEW: Inputs
const controlsArea = document.getElementById('controls-area');
const patientNameInput = document.getElementById('patient-name');
const startAnalysisBtn = document.getElementById('start-analysis-btn');
// REMOVED EMAIL SELECTOR

let selectedFile = null;

// Drag & Drop Logic
dropZone.addEventListener('click', (e) => {
    // Prevent clicking the dropzone if we are clicking inputs inside it
    if (e.target !== patientNameInput && e.target !== startAnalysisBtn) {
        fileInput.click();
    }
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#00d2ff';
    dropZone.style.background = 'rgba(0, 210, 255, 0.05)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'rgba(255,255,255,0.1)';
    dropZone.style.background = 'transparent';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255,255,255,0.1)';
    dropZone.style.background = 'transparent';
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file (JPG/PNG).');
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        document.querySelector('.drop-content').classList.add('hidden');
        previewContainer.classList.remove('hidden');
        
        // Show external controls
        controlsArea.classList.remove('hidden');
        loadingOverlay.classList.add('hidden');
        resultsCard.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// REMOVED VALIDATE EMAIL FUNCTION

// Start Analysis Button Click
startAnalysisBtn.addEventListener('click', () => {
    if (!selectedFile) return;

    // REMOVED EMAIL VALIDATION LOGIC

    let patientName = patientNameInput.value.trim();
    if (!patientName) {
        const randomID = Math.floor(1000 + Math.random() * 9000);
        patientName = `ANONYMOUS-${randomID}`;
    }

    // Hide Inputs, Show Loading
    controlsArea.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');
    
    simulateProcessSteps(selectedFile, patientName);
});

async function simulateProcessSteps(file, patientName) {
    const steps = [
        "Preprocessing Image...",
        "Checking for Blur...",
        "Extracting Visual Features (ResNet50)...",
        "Matching Semantic Text (CLIP)...",
        "Finalizing Diagnosis..."
    ];

    for (const step of steps) {
        loadingText.innerText = step;
        await new Promise(r => setTimeout(r, 600));
    }

    uploadAndAnalyze(file, patientName);
}

async function uploadAndAnalyze(file, patientName) {
    const formData = new FormData();
    formData.append('file', file);

    try {
        const startTime = Date.now();
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) throw new Error(data.error || "Server Error");

        const endTime = Date.now();
        const timeTaken = ((endTime - startTime) / 1000).toFixed(2);

        data.patient_name = patientName;
        displayResults(data, timeTaken);
    } catch (error) {
        loadingText.innerText = `Error: ${error.message}`;
        loadingText.style.color = "#ff3366";
    }
}

let currentResult = null;

function displayResults(data, timeTaken) {
    currentResult = data;
    loadingOverlay.classList.add('hidden');
    resultsCard.classList.remove('hidden');

    if (data.is_blurry) {
        document.getElementById('blur-warning').classList.remove('hidden');
    } else {
        document.getElementById('blur-warning').classList.add('hidden');
    }

    document.getElementById('patient-display').innerText = `Patient: ${data.patient_name}`;
    const predLabel = document.getElementById('pred-label');
    predLabel.innerText = data.label;
    
    if (data.label === "PNEUMONIA") {
        predLabel.style.color = "#ff3366";
        document.documentElement.style.setProperty('--accent', '#ff3366');
    } else {
        predLabel.style.color = "#00ff9d";
        document.documentElement.style.setProperty('--accent', '#00ff9d');
    }

    setTimeout(() => {
        document.getElementById('conf-bar').style.width = data.confidence + "%";
        document.getElementById('conf-val').innerText = data.confidence + "%";
    }, 100);

    document.getElementById('sharp-val').innerText = data.sharpness;
    document.getElementById('time-val').innerText = timeTaken + "s";
}

document.getElementById('download-btn').addEventListener('click', async () => {
    if (!currentResult) return;
    const btn = document.getElementById('download-btn');
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
    
    try {
        const response = await fetch('/report', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(currentResult)
        });
        const data = await response.json();
        window.open(data.url, '_blank');
        btn.innerHTML = '<i class="fas fa-check"></i> Report Downloaded';
    } catch (e) {
        btn.innerHTML = '<i class="fas fa-times"></i> Error';
    }
    
    setTimeout(() => {
        btn.innerHTML = '<i class="fas fa-file-download"></i> Report';
    }, 3000);
});