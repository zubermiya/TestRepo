// Main JavaScript for AI Media Detection

let currentFileName = '';
let currentFileType = '';

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const detailedAnalysisBtn = document.getElementById('detailedAnalysisBtn');

    // File input change event
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Detailed analysis button
    detailedAnalysisBtn.addEventListener('click', runDetailedAnalysis);

    // Click to upload
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadArea').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadArea').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('uploadArea').classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    // Validate file type
    const allowedTypes = [
        'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/tiff',
        'video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/wmv', 'video/flv', 'video/webm'
    ];

    if (!allowedTypes.includes(file.type)) {
        showError('Unsupported file type. Please upload an image or video file.');
        return;
    }

    // Check file size (100MB limit)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size too large. Please upload a file smaller than 100MB.');
        return;
    }

    currentFileName = file.name;
    currentFileType = file.type.startsWith('image/') ? 'image' : 'video';

    // Show loading modal
    showLoadingModal();

    // Upload and analyze file
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show progress
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    progressContainer.style.display = 'block';

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        hideLoadingModal();
        progressContainer.style.display = 'none';
        
        if (data.success) {
            displayResults(data);
            showMediaPreview(file, data.filename);
        } else {
            showError(data.error || 'Analysis failed');
        }
    })
    .catch(error => {
        hideLoadingModal();
        progressContainer.style.display = 'none';
        showError('Error uploading file: ' + error.message);
        console.error('Upload error:', error);
    });

    // Simulate progress for better UX
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressBar.style.width = progress + '%';
    }, 200);

    // Clear interval when done
    setTimeout(() => {
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
    }, 3000);
}

function displayResults(data) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    const detailedSection = document.getElementById('detailedSection');
    const metadataSection = document.getElementById('metadataSection');
    
    resultsSection.style.display = 'block';
    detailedSection.style.display = 'block';
    metadataSection.style.display = 'block';
    
    resultsSection.classList.add('fade-in');

    // Update file name
    document.getElementById('fileName').textContent = data.filename;

    // Update AI probability
    const aiProbability = Math.round(data.ai_probability * 100);
    const aiProbabilityElement = document.getElementById('aiProbability');
    const aiProbabilityBar = document.getElementById('aiProbabilityBar');
    
    aiProbabilityElement.textContent = aiProbability + '%';
    aiProbabilityBar.style.width = aiProbability + '%';
    
    // Color code based on probability
    if (aiProbability < 30) {
        aiProbabilityElement.className = 'badge bg-success fs-6';
        aiProbabilityBar.className = 'progress-bar low-risk';
    } else if (aiProbability < 70) {
        aiProbabilityElement.className = 'badge bg-warning fs-6';
        aiProbabilityBar.className = 'progress-bar medium-risk';
    } else {
        aiProbabilityElement.className = 'badge bg-danger fs-6';
        aiProbabilityBar.className = 'progress-bar high-risk';
    }

    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    document.getElementById('confidence').textContent = confidence + '%';
    document.getElementById('confidenceBar').style.width = confidence + '%';

    // Update verdict
    const verdict = document.getElementById('verdict');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDescription = document.getElementById('verdictDescription');
    
    verdictTitle.textContent = data.verdict;
    
    if (data.verdict === 'AI-Generated') {
        verdict.className = 'alert alert-danger';
        verdictDescription.textContent = 'This media appears to be artificially generated. Multiple strong indicators suggest AI involvement in its creation.';
    } else if (data.verdict === 'Likely AI-Generated') {
        verdict.className = 'alert alert-warning';
        verdictDescription.textContent = 'This media shows several indicators of AI generation, but the evidence is not conclusive.';
    } else if (data.verdict === 'Likely Real') {
        verdict.className = 'alert alert-success';
        verdictDescription.textContent = 'This media appears to be authentic. Analysis suggests it was created through traditional means.';
    } else if (data.verdict === 'Uncertain') {
        verdict.className = 'alert alert-info';
        verdictDescription.textContent = 'The analysis shows mixed indicators. The system cannot determine with confidence whether this is AI-generated or real.';
    } else {
        verdict.className = 'alert alert-warning';
        verdictDescription.textContent = 'Analysis could not be completed successfully. Please try again or contact support.';
    }

    // Update analysis factors
    const analysisFactors = document.getElementById('analysisFactors');
    analysisFactors.innerHTML = '';
    
    data.analysis.forEach(factor => {
        const li = document.createElement('li');
        li.textContent = factor;
        analysisFactors.appendChild(li);
    });

    // Display metadata
    displayMetadata(data.metadata);

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showMediaPreview(file, filename) {
    const mediaPreview = document.getElementById('mediaPreview');
    mediaPreview.innerHTML = '';

    if (currentFileType === 'image') {
        const img = document.createElement('img');
        img.src = URL.createObjectURL(file);
        img.alt = 'Uploaded image';
        img.className = 'img-fluid';
        mediaPreview.appendChild(img);
    } else if (currentFileType === 'video') {
        const video = document.createElement('video');
        video.src = URL.createObjectURL(file);
        video.controls = true;
        video.className = 'img-fluid';
        mediaPreview.appendChild(video);
    }
}

function displayMetadata(metadata) {
    const basicMetadata = document.getElementById('basicMetadata');
    const technicalMetadata = document.getElementById('technicalMetadata');
    
    basicMetadata.innerHTML = '';
    technicalMetadata.innerHTML = '';

    // Basic metadata
    const basicFields = ['mime_type', 'file_size', 'dimensions', 'format'];
    const technicalFields = ['exif', 'creation_time', 'modification_time'];

    basicFields.forEach(field => {
        if (metadata[field]) {
            const div = createMetadataItem(field, metadata[field]);
            basicMetadata.appendChild(div);
        }
    });

    technicalFields.forEach(field => {
        if (metadata[field]) {
            const div = createMetadataItem(field, metadata[field]);
            technicalMetadata.appendChild(div);
        }
    });

    // Video specific metadata
    if (currentFileType === 'video') {
        const videoFields = ['fps', 'frame_count', 'duration', 'width', 'height'];
        videoFields.forEach(field => {
            if (metadata[field]) {
                const div = createMetadataItem(field, metadata[field]);
                technicalMetadata.appendChild(div);
            }
        });
    }
}

function createMetadataItem(key, value) {
    const div = document.createElement('div');
    div.className = 'metadata-item';
    
    const keySpan = document.createElement('div');
    keySpan.className = 'metadata-key';
    keySpan.textContent = formatKey(key);
    
    const valueSpan = document.createElement('div');
    valueSpan.className = 'metadata-value';
    valueSpan.textContent = formatValue(value);
    
    div.appendChild(keySpan);
    div.appendChild(valueSpan);
    
    return div;
}

function formatKey(key) {
    return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatValue(value) {
    if (typeof value === 'object') {
        return JSON.stringify(value, null, 2);
    }
    if (typeof value === 'number') {
        if (value > 1000000) {
            return (value / 1000000).toFixed(2) + ' MB';
        }
        return value.toLocaleString();
    }
    return String(value);
}

function runDetailedAnalysis() {
    if (!currentFileName) {
        showError('No file uploaded for analysis');
        return;
    }

    const btn = document.getElementById('detailedAnalysisBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    btn.disabled = true;

    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            filename: currentFileName
        })
    })
    .then(response => response.json())
    .then(data => {
        displayDetailedResults(data);
        btn.innerHTML = originalText;
        btn.disabled = false;
    })
    .catch(error => {
        showError('Error running detailed analysis: ' + error.message);
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}

function displayDetailedResults(data) {
    const detailedResults = document.getElementById('detailedResults');
    
    let html = '<div class="analysis-grid">';
    
    // Compression analysis
    if (data.compression_analysis) {
        html += `
            <div class="analysis-card">
                <h6><i class="fas fa-compress me-2"></i>Compression Analysis</h6>
                <div class="analysis-value">${(data.compression_analysis.artifact_ratio * 100).toFixed(2)}%</div>
                <small class="text-muted">Artifact Ratio</small>
            </div>
        `;
    }

    // Image statistics analysis
    if (data.image_statistics) {
        const stats = data.image_statistics;
        if (stats.texture_variance !== undefined) {
            html += `
                <div class="analysis-card">
                    <h6><i class="fas fa-chart-bar me-2"></i>Texture Analysis</h6>
                    <div class="analysis-value">${stats.texture_variance.toFixed(0)}</div>
                    <small class="text-muted">Texture Variance</small>
                </div>
            `;
        }
        
        if (stats.frequency_energy !== undefined) {
            html += `
                <div class="analysis-card">
                    <h6><i class="fas fa-wave-square me-2"></i>Frequency Analysis</h6>
                    <div class="analysis-value">${stats.frequency_energy.toFixed(2)}</div>
                    <small class="text-muted">Frequency Energy</small>
                </div>
            `;
        }
        
        if (stats.ai_score !== undefined) {
            html += `
                <div class="analysis-card">
                    <h6><i class="fas fa-robot me-2"></i>AI Indicators</h6>
                    <div class="analysis-value">${(stats.ai_score * 100).toFixed(1)}%</div>
                    <small class="text-muted">AI Score</small>
                </div>
            `;
        }
    }

    // Face analysis
    if (data.face_analysis && data.face_analysis.faces_detected !== undefined) {
        html += `
            <div class="analysis-card">
                <h6><i class="fas fa-user me-2"></i>Face Detection</h6>
                <div class="analysis-value">${data.face_analysis.faces_detected}</div>
                <small class="text-muted">Faces Detected</small>
            </div>
        `;
    }

    // Video specific analysis
    if (data.frames_analyzed) {
        html += `
            <div class="analysis-card">
                <h6><i class="fas fa-film me-2"></i>Video Analysis</h6>
                <div class="analysis-value">${data.frames_analyzed}</div>
                <small class="text-muted">Frames Analyzed</small>
            </div>
        `;
    }

    // Detailed analysis
    if (data.detailed_analysis) {
        if (data.detailed_analysis.color_analysis) {
            const colorAnalysis = data.detailed_analysis.color_analysis;
            html += `
                <div class="analysis-card">
                    <h6><i class="fas fa-palette me-2"></i>Color Analysis</h6>
                    <div class="analysis-value">${colorAnalysis.color_variance.toFixed(2)}</div>
                    <small class="text-muted">Color Variance</small>
                </div>
            `;
        }
        
        if (data.detailed_analysis.motion_analysis) {
            const motionAnalysis = data.detailed_analysis.motion_analysis;
            html += `
                <div class="analysis-card">
                    <h6><i class="fas fa-running me-2"></i>Motion Analysis</h6>
                    <div class="analysis-value">${(motionAnalysis.motion_consistency * 100).toFixed(1)}%</div>
                    <small class="text-muted">Motion Consistency</small>
                </div>
            `;
        }
    }

    html += '</div>';
    
    // Add summary
    html += `
        <div class="mt-4">
            <h6>Analysis Summary</h6>
            <p class="text-muted">
                Detailed technical analysis completed. The above metrics provide insights into 
                various aspects of the media file that can indicate AI generation patterns.
            </p>
        </div>
    `;
    
    detailedResults.innerHTML = html;
}

function showLoadingModal() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

function hideLoadingModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

function showError(message) {
    // Create error alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        <strong>Error:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at top of container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}