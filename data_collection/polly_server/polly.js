const express = require('express');
const { PollyClient, SynthesizeSpeechCommand } = require('@aws-sdk/client-polly');

const app = express();
const port = 3000;

app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*'); // Replace '*' with your allowed client origin
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

app.get('/convert-to-speech', async (req, res) => {
  const text = req.query.text;
  console.log("got a request for:", text)

  // Configure AWS Polly with your credentials
  const client = new PollyClient({ region: 'us-east-1' });

  // Define the parameters for speech synthesis
  const command = new SynthesizeSpeechCommand({
    OutputFormat: 'mp3',
    Text: text,
    VoiceId: 'Joanna',
    Engine: 'neural',
  });

  try {
    // Use Polly to synthesize speech
    const response = await client.send(command);

    // Set the content type to audio/mpeg
    res.set('Content-Type', 'audio/mpeg');

    // Pipe the audio stream to the response
    response.AudioStream.pipe(res);
  } catch (error) {
    console.error('Error converting text to speech:', error);
    res.status(500).send('Error converting text to speech');
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
