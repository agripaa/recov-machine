const express = require('express');
const multer = require('multer');
const moment = require('moment');
const path = require('path');
const fs = require('fs');
const knex = require('knex')(require('./knexfile').development); // MySQL configuration

const app = express();
const PORT = process.env.PORT || 8000;
app.use(express.json());

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const dir = './uploads';
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir);
    }
    cb(null, dir);
  },
  filename: function (req, file, cb) {
    const timestamp = Date.now();
    cb(null, `${file.fieldname}_${timestamp}${path.extname(file.originalname)}`);
  },
});

const upload = multer({ storage: storage });

// Endpoint to save detection data
app.post('/api/detections', upload.single('capture'), async (req, res) => {
  const { type, date } = req.body;
  const file = req.file;

  if (!type || !date || !file) {
    return res.status(400).json({ error: 'Please provide type, date, and capture file.' });
  }

  try {
    const detection = {
      type,
      date: moment(date).format('YYYY-MM-DD'),
      capture: file.path,
    };

    const [id] = await knex('detections').insert(detection);
    res.status(201).json({ message: 'Detection data saved successfully!', data: { id, ...detection } });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to save data' });
  }
});

// Endpoint to fetch all detections
app.get('/api/detections', async (req, res) => {
  try {
    const detections = await knex('detections').select('*');
    res.json(detections);
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Failed to fetch data' });
  }
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
