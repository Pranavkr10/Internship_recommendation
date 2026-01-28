function toggleChat() {
    const box = document.getElementById("chat-box");

    if (!box) {
        console.error("chat-box not found");
        return;
    }

    if (box.style.display === "flex") {
        box.style.display = "none";
    } else {
        box.style.display = "flex";
    }
}

function sendMessage() {
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;

    const chatBody = document.getElementById("chat-body");

    // User message
    const userMsg = document.createElement("div");
    userMsg.className = "user-msg";
    userMsg.innerText = message;
    chatBody.appendChild(userMsg);

    input.value = "";

    // Show loading
    const loadingMsg = document.createElement("div");
    loadingMsg.className = "bot-msg";
    loadingMsg.innerText = "Thinking...";
    chatBody.appendChild(loadingMsg);

    fetch("/api/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message })
    })
    .then(res => {
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        // Remove loading message
        chatBody.removeChild(loadingMsg);
        
        const botMsg = document.createElement("div");
        botMsg.className = "bot-msg";

        if (data.error) {
            botMsg.innerText = "Error: " + data.error;
        } else {
            let responseText = data.summary;
            if (data.top_recommendation) {
                responseText += `\n\nTop recommendation: ${data.top_recommendation[1]?.title || 'Unknown'} (Score: ${data.top_recommendation[0]?.toFixed(2) || 0})`;
            }
            botMsg.innerText = responseText;
        }

        chatBody.appendChild(botMsg);
        chatBody.scrollTop = chatBody.scrollHeight;
    })
    .catch(error => {
        chatBody.removeChild(loadingMsg);
        const errorMsg = document.createElement("div");
        errorMsg.className = "bot-msg";
        errorMsg.innerText = "Error: " + error.message;
        chatBody.appendChild(errorMsg);
        chatBody.scrollTop = chatBody.scrollHeight;
        console.error("Chat error:", error);
    });
}