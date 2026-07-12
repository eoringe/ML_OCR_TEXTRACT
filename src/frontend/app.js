document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const workspace = document.getElementById("workspace");
    const sourceImage = document.getElementById("source-image");
    const bboxCanvas = document.getElementById("bbox-canvas");
    const btnToggleBoxes = document.getElementById("btn-toggle-boxes");
    const loadingOverlay = document.getElementById("loading-overlay");
    const loadingSteps = document.getElementById("loading-steps");
    const errorCard = document.getElementById("error-card");
    const errorMessage = document.getElementById("error-message");
    const btnRetry = document.getElementById("btn-retry");
    const btnDownloadJson = document.getElementById("btn-download-json");
    const btnCopyJson = document.getElementById("btn-copy-json");
    
    // Tab switching elements
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    // Results elements
    const resMerchant = document.getElementById("res-merchant");
    const resDate = document.getElementById("res-date");
    const resAddress = document.getElementById("res-address");
    const resTotal = document.getElementById("res-total");
    const resJsonBlock = document.getElementById("res-json-block");
    const resTranscriptsList = document.getElementById("res-transcripts-list");
    const transcriptSearch = document.getElementById("transcript-search");
    
    // Queue Elements
    const queueList = document.getElementById("queue-list");
    const queueCount = document.getElementById("queue-count");
    
    // Pipeline Flow Logs Element
    const flowStepsContainer = document.querySelector(".flow-steps");

    // Application State variables
    let queue = []; // Array of { file, filename, status, data, elapsedMs, previewUrl, errorMsg }
    let activeItemIdx = -1;
    let apiData = null;
    let showBoxes = true;
    let hoveredBoxIndex = -1;
    let resizeObserver = null;
    let isProcessing = false;

    // API Server Location
    const API_URL = window.location.port === "8000" 
        ? "/upload" 
        : "http://localhost:8000/upload";

    // 1. Drag and Drop event handlers
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });
    
    ["dragleave", "dragend"].forEach(type => {
        dropZone.addEventListener(type, () => dropZone.classList.remove("dragover"));
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length) {
            handleBatchFileUpload(e.dataTransfer.files);
        }
    });
    
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            handleBatchFileUpload(fileInput.files);
        }
    });

    // 2. Tab switching logic
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            tabButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // 3. Retry button (retry failed files)
    btnRetry.addEventListener("click", () => {
        errorCard.style.display = "none";
        if (activeItemIdx !== -1) {
            const item = queue[activeItemIdx];
            item.status = "pending";
            item.errorMsg = null;
            updateQueueUI();
            processNextInQueue();
        }
    });

    // 4. Download JSON button
    btnDownloadJson.addEventListener("click", () => {
        if (!apiData) return;
        const blob = new Blob([JSON.stringify(apiData, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `receipt_ocr_${apiData.filename || "result"}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // 5. Copy JSON button
    btnCopyJson.addEventListener("click", () => {
        if (!apiData) return;
        const jsonText = JSON.stringify(apiData, null, 2);
        navigator.clipboard.writeText(jsonText).then(() => {
            showToast("JSON copied to clipboard!");
        }).catch(() => {
            const textArea = document.createElement("textarea");
            textArea.value = jsonText;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand("copy");
            document.body.removeChild(textArea);
            showToast("JSON copied to clipboard!");
        });
    });

    function showToast(message) {
        const toast = document.createElement("div");
        toast.className = "copy-toast";
        toast.innerText = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        }, 2200);
    }

    // 6. Batch Upload Logic
    function handleBatchFileUpload(filesList) {
        const files = Array.from(filesList);
        if (!files.length) return;

        console.log(`[Batch Upload] Selected ${files.length} file(s) for extraction queue.`);
        workspace.style.display = "block";
        workspace.scrollIntoView({ behavior: "smooth" });

        const previousQueueLength = queue.length;

        files.forEach(file => {
            const queueItem = {
                file: file,
                filename: file.name,
                status: "pending",
                data: null,
                elapsedMs: null,
                previewUrl: null,
                errorMsg: null
            };

            const reader = new FileReader();
            reader.onload = (e) => {
                queueItem.previewUrl = e.target.result;
                const currentIdx = queue.indexOf(queueItem);
                if (currentIdx === activeItemIdx) {
                    sourceImage.src = queueItem.previewUrl;
                }
            };
            reader.readAsDataURL(file);

            queue.push(queueItem);
            console.log(`[Queue] Queued: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`);
        });

        // Auto-select first newly added item
        if (activeItemIdx === -1) {
            selectQueueItem(previousQueueLength);
        } else {
            updateQueueUI();
        }

        processNextInQueue();
    }

    function processNextInQueue() {
        if (isProcessing) return;

        const nextItemIdx = queue.findIndex(item => item.status === "pending");
        if (nextItemIdx === -1) {
            console.log("[Queue] All items in the queue have been processed.");
            return;
        }

        const item = queue[nextItemIdx];
        item.status = "processing";
        isProcessing = true;

        updateQueueUI();
        console.log(`[OCR Pipeline] STARTING processing for file: "${item.filename}"`);

        // If the processing item is currently selected, show the loading steps
        if (activeItemIdx === nextItemIdx) {
            showLoadingOverlay();
        }

        const formData = new FormData();
        formData.append("file", item.file);

        const startTime = Date.now();
        console.log(`[Network] Sending POST request to SROIE OCR API: ${API_URL}...`);

        fetch(API_URL, {
            method: "POST",
            body: formData
        })
        .then(response => {
            console.log(`[Network] Received response status ${response.status} for ${item.filename}`);
            if (!response.ok) {
                throw new Error(`Server returned error status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            item.status = "success";
            item.data = data;
            item.elapsedMs = Date.now() - startTime;
            
            console.log(`[OCR Pipeline] SUCCESS for "${item.filename}" in ${item.elapsedMs}ms.`);
            console.log(`[OCR Pipeline] Extracted Keys:`, {
                company: data.company,
                date: data.date,
                address: data.address,
                total: data.total
            });
            console.log(`[OCR Pipeline] Performance metrics (ms):`, data.steps_time_ms);
            
            if (activeItemIdx === nextItemIdx) {
                apiData = data;
                populateResults(data);
                setupCanvasDrawing();
            }
        })
        .catch(err => {
            console.error(`[OCR Pipeline] FAILED for "${item.filename}":`, err);
            item.status = "error";
            item.errorMsg = err.message;
            
            if (activeItemIdx === nextItemIdx) {
                showError(err.message);
            }
        })
        .finally(() => {
            isProcessing = false;
            if (activeItemIdx === nextItemIdx) {
                hideLoadingOverlay();
            }
            updateQueueUI();
            processNextInQueue();
        });
    }

    function selectQueueItem(idx) {
        if (idx < 0 || idx >= queue.length) return;
        activeItemIdx = idx;
        
        updateQueueUI();

        const activeItem = queue[idx];
        console.log(`[Queue UI] Selected item: "${activeItem.filename}" (Status: ${activeItem.status})`);
        errorCard.style.display = "none";
        workspace.style.display = "block";

        if (activeItem.previewUrl) {
            sourceImage.src = activeItem.previewUrl;
        } else {
            sourceImage.src = "";
        }

        if (activeItem.status === "processing") {
            showLoadingOverlay();
            clearResultsView();
        } else {
            hideLoadingOverlay();
        }

        if (activeItem.status === "success") {
            apiData = activeItem.data;
            populateResults(activeItem.data);
            setupCanvasDrawing();
        } else if (activeItem.status === "error") {
            showError(activeItem.errorMsg);
            clearResultsView();
        } else if (activeItem.status === "pending") {
            clearResultsView();
        }
    }

    function updateQueueUI() {
        queueCount.innerText = queue.length;
        queueList.innerHTML = "";

        queue.forEach((item, idx) => {
            const li = document.createElement("li");
            li.className = `queue-item ${idx === activeItemIdx ? 'active' : ''}`;
            
            let statusIcon = '<i class="fa-regular fa-clock status-pending"></i>';
            if (item.status === "processing") {
                statusIcon = '<i class="fa-solid fa-spinner fa-spin status-processing"></i>';
            } else if (item.status === "success") {
                statusIcon = '<i class="fa-solid fa-circle-check status-success"></i>';
            } else if (item.status === "error") {
                statusIcon = '<i class="fa-solid fa-circle-exclamation status-error"></i>';
            }

            const timeStr = item.elapsedMs ? `${(item.elapsedMs / 1000).toFixed(1)}s` : '';

            li.innerHTML = `
                <div class="queue-item-header">
                    <span class="queue-item-name" title="${item.filename}">${item.filename}</span>
                    <span class="queue-status">${statusIcon}</span>
                </div>
                <div class="queue-item-meta">
                    <span class="queue-time">${timeStr}</span>
                </div>
            `;

            li.addEventListener("click", () => selectQueueItem(idx));
            queueList.appendChild(li);
        });
    }

    function clearResultsView() {
        resMerchant.innerText = "-";
        resDate.innerText = "-";
        resAddress.innerText = "-";
        resTotal.innerText = "$0.00";
        resJsonBlock.innerText = "{}";
        resTranscriptsList.innerHTML = "";
        apiData = null;
        if (resizeObserver) {
            resizeObserver.disconnect();
            resizeObserver = null;
        }
        const ctx = bboxCanvas.getContext("2d");
        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
        
        // Reset pipeline stepper logs to base state
        flowStepsContainer.innerHTML = `
            <div class="flow-step"><i class="fa-regular fa-circle"></i> Task 1: Preprocessing</div>
            <div class="flow-step"><i class="fa-regular fa-circle"></i> Task 2: Localization</div>
            <div class="flow-step"><i class="fa-regular fa-circle"></i> Task 3: Text OCR</div>
            <div class="flow-step"><i class="fa-regular fa-circle"></i> Task 4: KIE Classification</div>
            <div class="flow-step"><i class="fa-regular fa-circle"></i> Task 5: Post-processing</div>
        `;
    }

    function showLoadingOverlay() {
        loadingOverlay.style.display = "flex";
        const steps = loadingSteps.querySelectorAll(".step-indicator");
        steps.forEach(step => step.classList.remove("active", "done"));
        
        let currentStep = 0;
        const stepInterval = setInterval(() => {
            if (currentStep > 0) {
                steps[currentStep - 1].classList.remove("active");
                steps[currentStep - 1].classList.add("done");
            }
            if (currentStep < steps.length) {
                steps[currentStep].classList.add("active");
                currentStep++;
            } else {
                clearInterval(stepInterval);
            }
        }, 850);
        
        loadingOverlay._stepInterval = stepInterval;
    }

    function hideLoadingOverlay() {
        if (loadingOverlay._stepInterval) {
            clearInterval(loadingOverlay._stepInterval);
        }
        loadingOverlay.style.display = "none";
    }

    function showError(message) {
        const friendlyMsg = message.includes("Failed to fetch") || message.includes("NetworkError")
            ? "Could not connect to the backend server. Make sure the API is running on port 8000 (python src/backend/app.py)."
            : `Pipeline error: ${message}`;

        errorMessage.innerText = friendlyMsg;
        errorCard.style.display = "block";
    }

    // 7. Populate Results into tabs, KIE card, and pipeline logs
    function populateResults(data) {
        resMerchant.innerText = data.company || "N/A";
        resDate.innerText = data.date || "N/A";
        resAddress.innerText = data.address || "N/A";
        resTotal.innerText = data.total ? `$${data.total}` : "N/A";
        
        resJsonBlock.innerText = JSON.stringify(data, null, 2);
        
        // Populate Transcripts List
        resTranscriptsList.innerHTML = "";
        if (data.transcripts) {
            data.transcripts.forEach((text, index) => {
                const li = document.createElement("li");
                li.className = "transcript-item";
                li.innerText = `${index + 1}. [Box ${index}] ${text}`;
                
                li.addEventListener("mouseenter", () => {
                    hoveredBoxIndex = index;
                    drawCanvasBoxes();
                });
                li.addEventListener("mouseleave", () => {
                    hoveredBoxIndex = -1;
                    drawCanvasBoxes();
                });
                
                resTranscriptsList.appendChild(li);
            });
        }

        // Render actual step durations inside logs stepper
        if (data.steps_time_ms) {
            const steps = data.steps_time_ms;
            flowStepsContainer.innerHTML = `
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 1: Preprocessing (${steps.preprocessing}ms)</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 2: Localization (${steps.detection}ms)</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 3: Text OCR (${steps.recognition}ms)</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 4: KIE Classification (${steps.kie}ms)</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 5: Post-processing (${steps.parsing}ms)</div>
            `;
        } else {
            flowStepsContainer.innerHTML = `
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 1: Preprocessing</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 2: Localization</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 3: Text OCR</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 4: KIE Classification</div>
                <div class="flow-step completed"><i class="fa-solid fa-circle-check"></i> Task 5: Post-processing</div>
            `;
        }
    }

    // Search transcripts function
    transcriptSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        const items = resTranscriptsList.getElementsByClassName("transcript-item");
        
        Array.from(items).forEach(item => {
            const match = item.innerText.toLowerCase().includes(query);
            item.style.display = match ? "block" : "none";
        });
    });

    // 8. Canvas Drawing & Bounding Box Coordinates Lookup
    function setupCanvasDrawing() {
        if (resizeObserver) {
            resizeObserver.disconnect();
        }

        if (sourceImage.complete) {
            syncCanvasSize();
        } else {
            sourceImage.onload = syncCanvasSize;
        }

        resizeObserver = new ResizeObserver(() => syncCanvasSize());
        resizeObserver.observe(sourceImage);
        
        bboxCanvas.addEventListener("mousemove", handleCanvasMouseMove);
        bboxCanvas.addEventListener("mouseleave", () => {
            hoveredBoxIndex = -1;
            drawCanvasBoxes();
        });
    }

    function syncCanvasSize() {
        bboxCanvas.width = sourceImage.clientWidth;
        bboxCanvas.height = sourceImage.clientHeight;
        drawCanvasBoxes();
    }

    function getScaleFactors() {
        const naturalW = sourceImage.naturalWidth || 640;
        const naturalH = sourceImage.naturalHeight || 640;
        const clientW = sourceImage.clientWidth || 640;
        const clientH = sourceImage.clientHeight || 640;
        
        return {
            scaleX: clientW / naturalW,
            scaleY: clientH / naturalH
        };
    }

    function drawCanvasBoxes() {
        const ctx = bboxCanvas.getContext("2d");
        ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);
        
        if (!apiData || !apiData.boxes || !showBoxes) return;
        
        const { scaleX, scaleY } = getScaleFactors();
        
        apiData.boxes.forEach((box, index) => {
            const isHovered = index === hoveredBoxIndex;
            
            ctx.beginPath();
            ctx.moveTo(box[0] * scaleX, box[1] * scaleY);
            ctx.lineTo(box[2] * scaleX, box[3] * scaleY);
            ctx.lineTo(box[4] * scaleX, box[5] * scaleY);
            ctx.lineTo(box[6] * scaleX, box[7] * scaleY);
            ctx.closePath();
            
            if (isHovered) {
                ctx.strokeStyle = "hsl(190, 95%, 50%)"; // Neon cyan
                ctx.lineWidth = 3;
                ctx.fillStyle = "rgba(190, 95%, 50%, 0.15)";
                ctx.fill();
                
                ctx.font = "bold 12px sans-serif";
                ctx.fillStyle = "hsl(190, 95%, 50%)";
                ctx.fillText(`[${index}]`, box[0] * scaleX, (box[1] * scaleY) - 5);
            } else {
                ctx.strokeStyle = "hsla(263, 90%, 65%, 0.6)"; // Violet
                ctx.lineWidth = 1.5;
            }
            ctx.stroke();
        });
    }

    function handleCanvasMouseMove(e) {
        if (!apiData || !apiData.boxes) return;
        
        const rect = bboxCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const { scaleX, scaleY } = getScaleFactors();
        
        let foundIndex = -1;
        
        for (let i = 0; i < apiData.boxes.length; i++) {
            const box = apiData.boxes[i];
            const x1 = Math.min(box[0], box[6]) * scaleX;
            const x2 = Math.max(box[2], box[4]) * scaleX;
            const y1 = Math.min(box[1], box[3]) * scaleY;
            const y2 = Math.max(box[5], box[7]) * scaleY;
            
            if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
                foundIndex = i;
                break;
            }
        }
        
        if (foundIndex !== hoveredBoxIndex) {
            hoveredBoxIndex = foundIndex;
            drawCanvasBoxes();
            
            if (foundIndex !== -1) {
                const items = resTranscriptsList.getElementsByClassName("transcript-item");
                if (items[foundIndex]) {
                    Array.from(items).forEach(it => it.style.borderColor = "var(--border-color)");
                    items[foundIndex].style.borderColor = "var(--secondary)";
                    items[foundIndex].scrollIntoView({ behavior: "smooth", block: "nearest" });
                }
            }
        }
    }

    // Toggle box button
    btnToggleBoxes.addEventListener("click", () => {
        showBoxes = !showBoxes;
        btnToggleBoxes.classList.toggle("btn-active", showBoxes);
        drawCanvasBoxes();
    });
});
