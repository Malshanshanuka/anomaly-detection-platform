require('dotenv').config();
const express = require('express');
const cors = require('cors');

// MongoDB Connection File එක සම්බන්ධ කිරීම (මෙය තිබිය යුත්තේ එක් වරක් පමණි)
const connectDB = require('./config/db');

// Initialize the Express application
const app = express();

// ==========================================
// 1. Database Connection
// ==========================================
connectDB();

// ==========================================
// 2. Middleware Setup
// ==========================================
// Allow cross-origin requests from our future Next.js frontend
app.use(cors()); 
// Parse incoming JSON requests
app.use(express.json());

// ==========================================
// 3. Basic Routes (API Endpoints)
// ==========================================
app.get('/', (req, res) => {
    res.status(200).json({
        service: "Node.js API Gateway",
        status: "Online",
        message: "Welcome to the AI Anomaly Detection Platform API"
    });
});

app.get('/api/health', (req, res) => {
    res.status(200).json({
        db_connected: true, 
        ai_engine_reachable: false, // We will update this later via Axios
        timestamp: new Date()
    });
});

// ==========================================
// 4. Server Initialization
// ==========================================
const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`\n=========================================`);
    console.log(` Node.js Backend is running on port ${PORT}`);
    console.log(`=========================================\n`);
});