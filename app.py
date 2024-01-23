import gradio as gr
from gradio_client import Client
import json
import re

def get_caption(image_in):
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

def get_magnet(prompt):
    amended_prompt = f"High quality sound effects. {prompt}"
    print(amended_prompt)
    client = Client("https://fffiloni-magnet.hf.space/--replicas/oo8sb/")
    result = client.predict(
        "facebook/audio-magnet-small",	# Literal['facebook/magnet-small-10secs', 'facebook/magnet-medium-10secs', 'facebook/magnet-small-30secs', 'facebook/magnet-medium-30secs', 'facebook/audio-magnet-small', 'facebook/audio-magnet-medium']  in 'Model' Radio component
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
    return result[0]['video']

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
    return result

def infer(image_in):
    caption = get_caption(image_in)
    magnet_result = get_magnet(caption)
    audioldm_result = get_audioldm(caption)
    return magnet_result, audioldm_result

css="""
#col-container{
    margin: 0 auto;
    max-width: 720px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <h2 style="text-align: center;">
            Image to SFX
        </h2>
        <p style="text-align: center;">
            Compare MAGNet and AudioLDM2 sound effects generation from image caption (Kosmos2)
        </p>
        """)
        
        with gr.Column():
            image_in = gr.Image(sources=["upload"], type="filepath", label="Image input", value="oiseau.png")
            submit_btn = gr.Button("Submit")
        with gr.Row():
            magnet_o = gr.Video(label="MAGNet output")
            audioldm2_o = gr.Video(label="AudioLDM2 output")
    submit_btn.click(
        fn=infer,
        inputs=[image_in],
        outputs=[magnet_o, audioldm2_o]
    )
demo.queue(max_size=10).launch(debug=True)