import gradio as gr
from gradio_client import Client
import os
import json
import re
from moviepy.editor import VideoFileClip
from moviepy.audio.AudioClip import AudioClip

hf_token = os.environ.get("HF_TKN")

def extract_audio(video_in):
    input_video = video_in
    output_audio = 'audio.wav'
    
    # Open the video file and extract the audio
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio
    
    # Save the audio as a .wav file
    audio_clip.write_audiofile(output_audio, fps=44100)  # Use 44100 Hz as the sample rate for .wav files  
    print("Audio extraction complete.")

    return 'audio.wav'

def get_caption_from_kosmos(image_in):
    kosmos2_client = Client("https://ydshieh-kosmos-2.hf.space/")

    kosmos2_result = kosmos2_client.predict(
        image_in,	# str (filepath or URL to image) in 'Test Image' Image component
        "Detailed",	# str in 'Description Type' Radio component
        fn_index=4
    )

    print(f"KOSMOS2 RETURNS: {kosmos2_result}")

    with open(kosmos2_result[1], 'r') as f:
        data = json.load(f)
    
    reconstructed_sentence = []
    for sublist in data:
        reconstructed_sentence.append(sublist[0])

    full_sentence = ' '.join(reconstructed_sentence)
    #print(full_sentence)

    # Find the pattern matching the expected format ("Describe this image in detail:" followed by optional space and then the rest)...
    pattern = r'^Describe this image in detail:\s*(.*)$'
    # Apply the regex pattern to extract the description text.
    match = re.search(pattern, full_sentence)
    if match:
        description = match.group(1)
        print(description)
    else:
        print("Unable to locate valid description.")

    # Find the last occurrence of "."
    last_period_index = description.rfind('.')

    # Truncate the string up to the last period
    truncated_caption = description[:last_period_index + 1]

    # print(truncated_caption)
    print(f"\n—\nIMAGE CAPTION: {truncated_caption}")
    
    return truncated_caption

def get_caption(image_in):
    client = Client("https://fffiloni-moondream1.hf.space/", hf_token=hf_token)
    result = client.predict(
		image_in,	# filepath  in 'image' Image component
		"Describe precisely the image in one sentence.",	# str  in 'Question' Textbox component
		#api_name="/answer_question"
        api_name="/predict"
    )
    print(result)
    return result

def get_magnet(prompt):
    amended_prompt = f"{prompt}"
    print(amended_prompt)
    client = Client("https://fffiloni-magnet.hf.space/")
    result = client.predict(
        "facebook/audio-magnet-medium",	# Literal['facebook/magnet-small-10secs', 'facebook/magnet-medium-10secs', 'facebook/magnet-small-30secs', 'facebook/magnet-medium-30secs', 'facebook/audio-magnet-small', 'facebook/audio-magnet-medium']  in 'Model' Radio component
        "",	# str  in 'Model Path (custom models)' Textbox component
        amended_prompt,	# str  in 'Input Text' Textbox component
        3,	# float  in 'Temperature' Number component
        0.9,	# float  in 'Top-p' Number component
        10,	# float  in 'Max CFG coefficient' Number component
        1,	# float  in 'Min CFG coefficient' Number component
        20,	# float  in 'Decoding Steps (stage 1)' Number component
        10,	# float  in 'Decoding Steps (stage 2)' Number component
        10,	# float  in 'Decoding Steps (stage 3)' Number component
        10,	# float  in 'Decoding Steps (stage 4)' Number component
        "prod-stride1 (new!)",	# Literal['max-nonoverlap', 'prod-stride1 (new!)']  in 'Span Scoring' Radio component
        api_name="/predict_full"
    )
    print(result)
    return result[1]

def get_audioldm(prompt):
    client = Client("https://haoheliu-audioldm2-text2audio-text2music.hf.space/")
    result = client.predict(
        prompt,	# str in 'Input text' Textbox component
        "Low quality. Music.",	# str in 'Negative prompt' Textbox component
        10,	# int | float (numeric value between 5 and 15) in 'Duration (seconds)' Slider component
        3.5,	# int | float (numeric value between 0 and 7) in 'Guidance scale' Slider component
        45,	# int | float in 'Seed' Number component
        3,	# int | float (numeric value between 1 and 5) in 'Number waveforms to generate' Slider component
        fn_index=1
    )
    print(result)
    audio_result = extract_audio(result)
    return audio_result

def get_audiogen(prompt):
    client = Client("https://fffiloni-audiogen.hf.space/")
    result = client.predict(
        prompt,
        10,
        api_name="/infer"
    )
    return result

def get_tango(prompt):
    try:
        client = Client("https://declare-lab-tango.hf.space/")
    except:
        raise gr.Error("Tango space API is not ready, please try again in few minutes ")
    
    result = client.predict(
				prompt,	# str representing string value in 'Prompt' Textbox component
				100,	# int | float representing numeric value between 100 and 200 in 'Steps' Slider component
				4,	# int | float representing numeric value between 1 and 10 in 'Guidance Scale' Slider component
				api_name="/predict"
    )
    print(result)
    return result

def infer(image_in, chosen_model):
    caption = get_caption(image_in)
    if chosen_model == "MAGNet" :
        magnet_result = get_magnet(caption)
        return magnet_result
    elif chosen_model == "AudioLDM-2" : 
        audioldm_result = get_audioldm(caption)
        return audioldm_result
    elif chosen_model == "AudioGen" :
        audiogen_result = get_audiogen(caption)
        return audiogen_result
    elif chosen_model == "Tango" :
        tango_result = get_tango(caption)
        return tango_result

css="""
#col-container{
    margin: 0 auto;
    max-width: 800px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <h2 style="text-align: center;">
            Image to SFX
        </h2>
        <p style="text-align: center;">
            Compare MAGNet, AudioLDM2 and AudioGen sound effects generation from image caption.
        </p>
        """)
        
        with gr.Column():
            image_in = gr.Image(sources=["upload"], type="filepath", label="Image input", value="oiseau.png")
            with gr.Row():
                chosen_model = gr.Dropdown(label="Choose a model", choices=["MAGNet", "AudioLDM-2", "AudioGen", "Tango"], value="AudioLDM-2")
                submit_btn = gr.Button("Submit")
        with gr.Column():
            audio_o = gr.Audio(label="Audio output")
    
    submit_btn.click(
        fn=infer,
        inputs=[image_in, chosen_model],
        outputs=[audio_o],
        concurrency_limit = 2
    )

demo.queue(max_size=10).launch(debug=True)