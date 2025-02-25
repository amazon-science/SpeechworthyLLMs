import boto3
from speechllm.utils import create_audio_file

# Example usage
text_to_convert = 'Hello, how are you?'
output_s3_bucket = 'jcho-voicellm'
output_s3_key = 'data/audio/test.mp3'

def play_audio_with_polly(text, engine="neural", fn="test", text_type="text"): 

    # Create a session using your AWS credentials
    session = boto3.Session()
    polly = session.client('polly')

    voice_id = "Joanna"
    output_format = "mp3"

    response = polly.synthesize_speech(
        OutputFormat=output_format,
        Text=text,
        VoiceId=voice_id,
        Engine=engine, 
        TextType=text_type
    )

    # Save the audio from the response
    audio = response['AudioStream'].read()

    # Save the audio to a file
    with open(f'{fn}.mp3', 'wb') as file:
        file.write(audio)

    return 


if __name__ == "__main__": 
    text_basic = "<speak>heya buddy, what are you up to today? I'm about to go look at a rental property this afternoon</speak>"
    play_audio_with_polly(text_basic, engine="neural", fn="test_neural", text_type="ssml")
    play_audio_with_polly(text_basic, engine="standard", fn="test_standard", text_type="ssml")

    text_with_ssml = """
    <speak>
        <prosody rate="medium" volume="loud">
            <s>Heya buddy,</s>
            <s>what are <emphasis level="moderate">you</emphasis> up to today?</s>
        </prosody>
        <break time="500ms"/> 
        <prosody rate="medium" volume="loud">
            <s>I'm about to go look at a <emphasis level="strong">rental property</emphasis> this afternoon.</s>
        </prosody>
    </speak>
"""

    play_audio_with_polly(text_with_ssml, engine="standard", fn="test_standard_ssml", text_type="ssml")
