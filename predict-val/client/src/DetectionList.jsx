import React, { useState, useEffect } from 'react';
import axios from 'axios';

const DetectionsList = () => {
  const [detections, setDetections] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDetections = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/detections');
        setDetections(response.data);
      } catch (error) {
        setError('Could not fetch data. Please ensure the server is running.');
        console.error(error);
      }
    };

    fetchDetections();
  }, []);

  if (error) {
    return <p>{error}</p>;
  }

  return (
    <div>
      <h1>Detections List</h1>
      {detections.length === 0 ? (
        <p>No data available.</p>
      ) : (
        <ul>
          {detections.map((detection) => (
            <li key={detection.id}>
              <p>Type: {detection.type}</p>
              <p>Date: {detection.date}</p>
              <p>Image:</p>
              <img
                src={`http://localhost:8000/${detection.capture}`}
                alt="Detection capture"
                style={{ width: '200px', height: 'auto' }}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default DetectionsList;
