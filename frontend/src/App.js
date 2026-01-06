import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages([...messages, userMessage]);
    setInput("");

    try {
      const response = await axios.post("http://localhost:8000/api/chat", {
        msg: input
      });

      const botMessage = {
        role: "bot",
        content: response.data.answer
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
    }
  };

  return (
    <div className="chat-container">
      <h1>LegalGPT: Your Legal Assistant</h1>
      <h4>Please crosscheck responses, LegalGPT is an AI assistant and may not be entirely accurate sometimes.</h4>
      <div className="chat-box">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role}`}>
            <strong>{m.role === "user" ? "You" : "Bot"}:</strong> {m.content}
          </div>
        ))}
      </div>
      <div className="input-container">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about Indian Law..."
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;
