// Function to start the webcam
function startWebcam() {
    const video = document.getElementById('video');
    if (video) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
            })
            .catch(error => {
                console.error("Error accessing the webcam:", error);
            });
    } else {
        console.error("Video element not found.");
    }
}

// Call the startWebcam function once the DOM is fully loaded
document.addEventListener('DOMContentLoaded', (event) => {
    startWebcam();
});

// Function to start real-time detection
function startDetection() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('overlay');
    const ctx = canvas.getContext('2d');
    const realTimeOutputDiv = document.getElementById('realTimeOutput');

    setInterval(() => {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the captured frame to base64
        const imageData = canvas.toDataURL('image/png');

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        })
        .then(response => response.json())
        .then(data => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (data.prediction && data.bbox) {
                const { x, y, w, h } = data.bbox;

                // Draw bounding box
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, w, h);

                // Display prediction label
                ctx.font = '16px Arial';
                ctx.fillStyle = 'red';
                ctx.fillText(data.prediction, x, y - 10);

                realTimeOutputDiv.innerHTML = `Real-Time Prediction: ${data.prediction}`;
            } else {
                realTimeOutputDiv.innerHTML = 'No face detected';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }, 1000); // Capture and send frame every second
}

// Function to upload an image
function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const uploadOutputDiv = document.getElementById('uploadOutput');
    
    if (fileInput && fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.prediction) {
                uploadOutputDiv.innerHTML = `Uploaded Image Prediction: ${data.prediction}`;
            } else {
                uploadOutputDiv.innerHTML = 'No face detected';
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Please select an image before uploading.');
    }
}
