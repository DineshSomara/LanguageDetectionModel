import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [inputText, setInputText] = useState('');
  const [predictedLanguage, setPredictedLanguage] = useState('');

  const handlePredictLanguage = async () => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', { text: inputText });
      setPredictedLanguage(response.data.predicted_language);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  return (
    <div>
      <h1>Language Detection</h1>
      <input type="text" value={inputText} onChange={(e) => setInputText(e.target.value)} />
      <button onClick={handlePredictLanguage}>Predict Language</button>
      {predictedLanguage && <p>Predicted Language: {predictedLanguage}</p>}
    </div>
  );
}

export default App;