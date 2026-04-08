import React, { useState, useRef, useEffect } from "react";
import "./App.css";

const OPENAI_MODELS = [
  "gpt-4o-mini",
  "gpt-4.1-mini",
  "gpt-4.1",
  "gpt-4o",
];

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [activeIndexName, setActiveIndexName] = useState("");
  const [expandedContexts, setExpandedContexts] = useState({});
  const [modelProvider, setModelProvider] = useState("openai");
  const [modelName, setModelName] = useState("gpt-4o-mini");
  const [apiKey, setApiKey] = useState("");
  const isChatReady = Boolean(uploadedFileName);
  const hasValidModelConfig = modelProvider !== "openai" || Boolean(apiKey.trim());

  const chatMessagesRef = useRef(null);

  useEffect(() => {
    if (!chatMessagesRef.current || chatHistory.length === 0) return;
    chatMessagesRef.current.scrollTo({
      top: chatMessagesRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [chatHistory]);

  const toggleContext = (messageIndex) => {
    setExpandedContexts((prev) => ({
      ...prev,
      [messageIndex]: !prev[messageIndex],
    }));
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setMessage("");
    setQuestion("");
    setChatHistory([]);
    setActiveIndexName("");
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a PDF first.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (response.ok) {
        setMessage("PDF uploaded and processed successfully.");
        setUploadedFileName(file.name);
        setActiveIndexName(data.index_name || "");
        setChatHistory([]);
      } else {
        setMessage(`Upload failed: ${data.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setMessage("Error occurred while uploading.");
    } finally {
      setLoading(false);
    }
  };

  const handleAsk = async (e) => {
    e.preventDefault();
    if (!isChatReady || !question.trim() || isProcessing) return;
    if (modelProvider === "openai" && !apiKey.trim()) {
      setChatHistory((prev) => [
        ...prev,
        {
          type: "error",
          content: "Please enter your OpenAI API key before asking a question.",
        },
      ]);
      return;
    }

    setIsProcessing(true);
    const currentQuestion = question;
    setQuestion("");

    setChatHistory((prev) => [
      ...prev,
      {
        type: "user",
        content: currentQuestion,
      },
    ]);

    try {
      const response = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: currentQuestion,
          index_name: activeIndexName,
          model_provider: modelProvider,
          model_name: modelName,
          api_key: apiKey.trim(),
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setChatHistory((prev) => [
          ...prev,
          {
            type: "assistant",
            content: data.answer,
            context: data.retrieved_chunks,
          },
        ]);
      } else {
        setChatHistory((prev) => [
          ...prev,
          {
            type: "error",
            content: "Sorry, I could not process your question. Please try again.",
          },
        ]);
      }
    } catch (error) {
      console.error("Error asking question:", error);
      setChatHistory((prev) => [
        ...prev,
        {
          type: "error",
          content: "Network error occurred. Please try again.",
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={`container ${isDarkMode ? "dark" : "light"}`}>
      <button
        className="theme-toggle"
        onClick={() => setIsDarkMode(!isDarkMode)}
        aria-label="Toggle theme"
      >
        {isDarkMode ? "Light" : "Dark"}
      </button>
      <div className="upload-card">
        <h1>PDF Chat Assistant</h1>
        <p className="subtitle">
          Upload any PDF and ask questions about its content.
          I will help you find the information you need.
        </p>
        <div className="project-credit">
          <span className="credit-pill">Built by Nitish Jha</span>
          <p className="credit-text">
            Powered by Hybrid RAG to generate more accurate, context-grounded answers.
          </p>
        </div>
        <div className="model-config">
          <h3 className="config-title">Model Settings</h3>
          <div className="config-grid">
            <label className="config-field">
              <span className="config-label">Provider</span>
              <select
                className="config-input"
                value={modelProvider}
                onChange={(e) => setModelProvider(e.target.value)}
              >
                <option value="openai">OpenAI</option>
              </select>
            </label>
            <label className="config-field">
              <span className="config-label">Model</span>
              <select
                className="config-input"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
              >
                {OPENAI_MODELS.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </label>
            <label className="config-field config-field-full">
              <span className="config-label">OpenAI API Key</span>
              <input
                type="password"
                className="config-input"
                placeholder="sk-..."
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
              />
            </label>
          </div>
        </div>

        <div className="upload-section">
          <label className="file-input-label">
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              className="file-input"
            />
            <span className="file-input-text">
              {file ? file.name : "Drop your PDF here or click to browse"}
            </span>
          </label>

          <button
            onClick={handleUpload}
            className="upload-button"
            disabled={!file || loading}
          >
            {loading ? (
              <span className="loading">Processing PDF...</span>
            ) : (
              <span className="button-content">Upload PDF</span>
            )}
          </button>
        </div>

        <div className="status-slot">
          <div
            className={`message ${message ? "visible" : "hidden"} ${
              message.includes("successfully") ? "success" : "error"
            }`}
          >
            {message || "Status placeholder"}
          </div>
        </div>

        <div className={`chat-section ${isChatReady ? "active" : "inactive"}`}>
          <p className="active-file">
            {isChatReady ? (
              <>
                Currently chatting with: <strong>{uploadedFileName}</strong>
              </>
            ) : (
              "Upload a PDF to start chatting"
            )}
          </p>

          <div className="chat-messages" ref={chatMessagesRef}>
            {chatHistory.length === 0 && (
              <p className="chat-placeholder">
                {isChatReady
                  ? "Ask a question to begin the conversation."
                  : "Your conversation will appear here after upload."}
              </p>
            )}
            {chatHistory.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.type}`}>
                <div className="message-content">
                  {msg.type === "user" ? "User" : "Assistant"}: {msg.content}
                </div>
                {msg.context && (
                  <>
                    <button
                      className="context-toggle"
                      onClick={() => toggleContext(index)}
                    >
                      {expandedContexts[index] ? "Hide context" : "Show context"}
                    </button>
                    {expandedContexts[index] && (
                      <div className="context-content">
                        <h4>Source Context:</h4>
                        {msg.context.map((chunk, i) => (
                          <p key={i} className="context-chunk">
                            {chunk}
                          </p>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            ))}
          </div>

          <form onSubmit={handleAsk} className="question-box">
            <input
              type="text"
              placeholder={
                !isChatReady
                  ? "Upload a PDF to enable chat"
                  : !hasValidModelConfig
                    ? "Enter your OpenAI API key above to start asking"
                    : "Ask any question about your PDF..."
              }
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="question-input"
              disabled={!isChatReady || !hasValidModelConfig || isProcessing}
            />
            <button
              type="submit"
              className="ask-button"
              disabled={!isChatReady || !hasValidModelConfig || !question.trim() || isProcessing}
              aria-label="Send message"
            >
              {isProcessing ? (
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              ) : (
                "Ask"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default App;
