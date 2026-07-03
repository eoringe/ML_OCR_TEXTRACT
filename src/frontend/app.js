document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const workspace = document.getElementById("workspace");
    const sourceImage = document.getElementById("source-image");
    const bboxCanvas = document.getElementById("bbox-canvas");
    const btnToggleBoxes = document.getElementById("btn-toggle-boxes");
    
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
    
    // Application State variables
    let apiData = null;
    let showBoxes = true;
    let hoveredBoxIndex = -1;
    let resizeObserver = null;

    // API Server Location
    const API_URL = window.location.origin === "http://localhost:8000" || window.location.origin.includes("127.0.0.1") 
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
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            handleFileUpload(fileInput.files[0]);
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

    // 3. Upload and API Request Pipeline
    function handleFileUpload(file) {
        // Display local preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            sourceImage.src = e.target.result;
            workspace.style.display = "block";
            // Scroll to workspace smoothly
            workspace.scrollIntoView({ behavior: "smooth" });
        };
        reader.readAsDataURL(file);
        
        // Show Loading State in UI
        setLoadingState(true);
        
        // Prepare API request payload
        const formData = new FormData();
        formData.append("file", file);
        
        fetch(API_URL, {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server returned error status ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            apiData = data;
            populateResults(data);
            setupCanvasDrawing();
        })
        .catch(err => {
            console.error("OCR Pipeline extraction failed:", err);
            alert(`OCR pipeline error: ${err.message}. (Make sure the backend server python app.py is running on port 8000)`);
        })
        .finally(() => {
            setLoadingState(false);
        });
    }

    function setLoadingState(isLoading) {
        if (isLoading) {
            dropZone.style.pointerEvents = "none";
            dropZone.style.opacity = "0.7";
            dropZone.querySelector("h3").innerText = "Analyzing receipt images...";
            dropZone.querySelector("p").innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing through ML pipeline layers...';
        } else {
            dropZone.style.pointerEvents = "all";
            dropZone.style.opacity = "1";
            dropZone.querySelector("h3").innerText = "Drag & drop receipt image";
            dropZone.querySelector("p").innerHTML = 'or <span class="browse-link">browse files</span> from your computer';
        }
    }

    // 4. Populate Results into tabs and values
    function populateResults(data) {
        resMerchant.innerText = data.company || "N/A";
        resDate.innerText = data.date || "N/A";
        resAddress.innerText = data.address || "N/A";
        resTotal.innerText = data.total ? `$${data.total}` : "N/A";
        
        resJsonBlock.innerText = JSON.stringify(data, null, 2);
        
        // Populate Transcripts List
        resTranscriptsList.innerHTML = "";
        data.transcripts.forEach((text, index) => {
            const li = document.createElement("li");
            li.className = "transcript-item";
            li.innerText = `${index + 1}. [Box ${index}] ${text}`;
            
            // Sync hover transcript -> highlight canvas box
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

    // Search transcripts function
    transcriptSearch.addEventListener("input", (e) => {
        const query = e.target.value.toLowerCase();
        const items = resTranscriptsList.getElementsByClassName("transcript-item");
        
        Array.from(items).forEach(item => {
            const match = item.innerText.toLowerCase().includes(query);
            item.style.display = match ? "block" : "none";
        });
    });

    // 5. Canvas Drawing & Hover Tracking Coordinate Math
    function setupCanvasDrawing() {
        if (resizeObserver) {
            resizeObserver.disconnect();
        }

        // Handle image loading to configure dimensions
        if (sourceImage.complete) {
            syncCanvasSize();
        } else {
            sourceImage.onload = syncCanvasSize;
        }

        // Adjust canvas if wrapper or window resizes
        resizeObserver = new ResizeObserver(() => syncCanvasSize());
        resizeObserver.observe(sourceImage);
        
        // Setup mouse move coordinates lookup
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
        // Calculate scaling multipliers between natural image dims and CSS display size
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
            
            // Map SROIE 8-point coordinates [x1,y1,x2,y2,x3,y3,x4,y4] to client canvas
            ctx.beginPath();
            ctx.moveTo(box[0] * scaleX, box[1] * scaleY);
            ctx.lineTo(box[2] * scaleX, box[3] * scaleY);
            ctx.lineTo(box[4] * scaleX, box[5] * scaleY);
            ctx.lineTo(box[6] * scaleX, box[7] * scaleY);
            ctx.closePath();
            
            // Highlight styling
            if (isHovered) {
                ctx.strokeStyle = "hsl(190, 95%, 50%)"; // Neon cyan
                ctx.lineWidth = 3;
                ctx.fillStyle = "rgba(190, 95%, 50%, 0.15)";
                ctx.fill();
                
                // Draw floating text label
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
        
        // Find which bounding box matches the mouse coordinate
        // (Uses a simple point-in-polygon bounding box rectangle proxy for speed)
        for (let i = 0; i < apiData.boxes.length; i++) {
            const box = apiData.boxes[i];
            const x1 = Math.min(box[0], box[6]) * scaleX;
            const x2 = Math.max(box[2], box[4]) * scaleX;
            const y1 = Math.min(box[1], box[3]) * scaleY;
            const y2 = Math.max(box[5], box[7]) * scaleY;
            
            if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
                foundIndex = i;
                break; // Closest match
            }
        }
        
        if (foundIndex !== hoveredBoxIndex) {
            hoveredBoxIndex = foundIndex;
            drawCanvasBoxes();
            
            // Sync hover from canvas -> scroll and highlight transcript list item
            if (foundIndex !== -1) {
                const items = resTranscriptsList.getElementsByClassName("transcript-item");
                if (items[foundIndex]) {
                    // Remove other active lists
                    Array.from(items).forEach(it => it.style.borderColor = "var(--border-color)");
                    
                    // Highlight match
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
