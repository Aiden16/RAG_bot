import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage("");
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

      if (response.ok) {
        setMessage("✅ PDF uploaded and processed successfully!");
      } else {
        setMessage("❌ Upload failed.");
      }
    } catch (error) {
      console.error("Error uploading file:", error);
      setMessage("❌ Error occurred while uploading.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="upload-card">
        <h1>PDF Upload</h1>
        <p className="subtitle">Upload your PDF file for processing</p>
        
        <div className="upload-section">
          <label className="file-input-label">
            <input 
              type="file" 
              accept="application/pdf" 
              onChange={handleFileChange}
              className="file-input" 
            />
            <span className="file-input-text">
              {file ? file.name : "Choose a PDF file"}
            </span>
          </label>

          <button 
            onClick={handleUpload} 
            className="upload-button"
            disabled={!file || loading}
          >
            {loading ? "Uploading..." : "Upload PDF"}
          </button>
        </div>

        {message && (
          <div className={`message ${message.includes("✅") ? "success" : "error"}`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
