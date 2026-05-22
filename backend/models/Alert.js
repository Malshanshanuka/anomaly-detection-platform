const mongoose = require('mongoose');

const AlertSchema = new mongoose.Schema({
    severity: {
        type: String,
        required: true,
        enum: ['info', 'warning', 'critical'], // Only allow these specific values
        default: 'info'
    },
    message: {
        type: String,
        required: true
    },
    detail: {
        type: String,
        required: false // Optional field for extra context (e.g., "Node-05")
    },
    error_score: {
        type: Number,
        required: false // Optional, as 'info' logs might not have an AI score
    },
    is_anomaly: {
        type: Boolean,
        default: false
    },
    timestamp: {
        type: Date,
        default: Date.now // Automatically sets the exact time the alert is created
    }
});

// Create and export the Mongoose model
module.exports = mongoose.model('Alert', AlertSchema);