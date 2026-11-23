import re
import streamlit as st
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3.5:9b"


def ask_ollama(messages, temperature=0.7):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "think": False,
        "stream": False,
        "options": {"temperature": temperature},
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()
    return result["message"]["content"].strip()


def prompt_instructions(creation_type):
    if creation_type == "Image":
        return """
You are an expert AI image prompt director.
Create one polished image-generation prompt using the user's original idea
and any additional answers they supplied.

Rules:
- Write exactly one cohesive paragraph of 100–150 words.
- Preserve the original concept and every relevant detail from the answers.
- Improve composition, visual clarity, lighting, atmosphere, textures, depth,
  and artistic detail where appropriate.
- Do not invent unrelated characters, objects, or settings.
- This is a still image: do not describe animation, video, or camera movement.
- Return only the finished prompt, with no title, headings, bullets, or commentary.
"""
    return """
You are an expert AI video prompt director.
Create one polished video-generation prompt using the user's original idea
and any additional answers they supplied.

Rules:
- Write 120–180 words.
- Preserve the original concept and every relevant detail from the answers.
- Describe the subject's actions and movement.
- Include specific camera movement, such as a tracking shot, push-in, pan,
  tilt, dolly, or aerial follow.
- Include natural scene progression and pacing.
- Improve lighting, atmosphere, textures, and visual depth.
- Do not invent unrelated characters or settings.
- Return only the finished prompt, with no title, headings, or commentary.
"""


def quick_enhance(idea, creation_type):
    return ask_ollama(
        [
            {"role": "system", "content": prompt_instructions(creation_type)},
            {"role": "user", "content": idea},
        ],
        temperature=0.7,
    )

def creative_exploration(idea, creation_type):
    if creation_type == "Image":
        format_rules = """
Create three clearly different image-prompt directions:

1. Cinematic Realism
2. Stylized Artwork
3. Experimental Composition

BALANCED CREATIVITY PRINCIPLE:

Preserve the user's central concept while exploring imaginative
visual interpretations.

Every direction must retain:
- The original main subject.
- The original setting and environment.
- The essential concept and intended meaning.

You may creatively enhance:
- Lighting, atmosphere, and color palettes.
- Composition, perspective, and visual framing.
- Textures, materials, and environmental details.
- Artistic techniques and visual treatments.
- Subtle supporting details that strengthen the original concept.

CREATIVE LIMITS:

- Do not replace the original subject or setting.
- Do not introduce unrelated characters or major objects.
- Do not invent major events or change the underlying story.
- Do not add dramatic elements that fundamentally change the scene.
- Imaginative additions must enhance the original idea, not compete
  with or distract from it.
- Keep each direction visually coherent and suitable for an
  image-generation model.

MAKE THE THREE DIRECTIONS DISTINCT:

1. Cinematic Realism:
   Emphasize believable materials, realistic lighting, natural
   atmosphere, detailed environments, and cinematic composition.

2. Stylized Artwork:
   Explore a distinctive artistic treatment, such as painterly
   textures, illustration, stylized shapes, or imaginative colors.
   Keep the original subject and environment recognizable.

3. Experimental Composition:
   Explore unusual perspectives, framing, visual geometry,
   unconventional lighting, and creative composition.
   Experiment with HOW the original scene is presented, not
   with replacing its subject or inventing a different story.

IMAGE REQUIREMENTS:

- Each direction must contain one complete, cohesive image prompt.
- Do not describe animation, video, or subject movement.
- Describe a still visual composition.
- Avoid unnecessary technical specifications.
- Use clear, descriptive language suitable for image generation.
"""
    else:
        format_rules = """
Create three clearly different video-prompt directions:

1. Cinematic Realism
2. Stylized Visual Storytelling
3. Experimental Camera and Visual Approach

BALANCED CREATIVITY PRINCIPLE:

Preserve the user's central concept while exploring imaginative
visual interpretations.

Every direction must retain:
- The original main subject.
- The original setting and environment.
- The essential action or intended activity.
- The central meaning of the original idea.

You may creatively enhance:
- Subject movement and natural actions.
- Environmental movement and atmospheric effects.
- Lighting, colors, and visual style.
- Camera movement and framing.
- Scene progression, pacing, and cinematic detail.
- Subtle supporting elements that enhance the original concept.

CREATIVE LIMITS:

- Do not replace the original subject or setting.
- Do not invent unrelated characters or major objects.
- Do not introduce major events that change the original story.
- Do not give the subject unexpected actions that fundamentally
  change the user's intended concept.
- Do not add unnecessary destruction, transformations,
  dramatic accidents, or unrelated narrative twists.
- Imaginative additions must enhance the original idea.
- Keep the action and scene progression coherent and achievable
  for modern AI video-generation models.

MAKE THE THREE DIRECTIONS DISTINCT:

1. Cinematic Realism:
   Emphasize realistic subject movement, natural environmental
   behavior, cinematic lighting, controlled camera movement,
   and believable scene progression.

2. Stylized Visual Storytelling:
   Explore distinctive artistic treatments, stylized movement,
   imaginative lighting, expressive colors, and creative visual
   storytelling while preserving the original action and setting.

3. Experimental Camera and Visual Approach:
   Explore unusual perspectives, creative framing, interesting
   camera movements, visual transitions, and unconventional
   compositions.
   Experiment with HOW the original scene is filmed, not by
   replacing its subject or inventing an unrelated story.

VIDEO REQUIREMENTS:

- Every direction must describe the subject's movement or action.
- Every direction must include specific camera movement.
- Every direction must describe a logical progression through time.
- Camera movements should be clear, controlled, and achievable.
- Avoid excessive camera rotations, conflicting movements,
  or unnecessarily complicated transitions.
- Maintain continuity between the subject, environment,
  and camera throughout the scene.
"""

    system_instructions = (
        "You are an expert AI creative director specializing in "
        "high-quality AI image and video prompts.\n\n"
        + format_rules
        + """

OUTPUT FORMAT:

Return exactly three numbered creative directions.

For each direction, provide:
- A clearly labeled direction name.
- A short descriptive heading.
- One complete, polished generation prompt.

Use this structure:

### 1. [Direction Name]
**Heading:** [Short descriptive heading]
**Prompt:** [Complete generation prompt]

### 2. [Direction Name]
**Heading:** [Short descriptive heading]
**Prompt:** [Complete generation prompt]

### 3. [Direction Name]
**Heading:** [Short descriptive heading]
**Prompt:** [Complete generation prompt]

Return all three directions and nothing else.
Do not include an introduction, conclusion, or commentary.
"""
    )

    return ask_ollama(
        [
            {"role": "system", "content": system_instructions},
            {
                "role": "user",
                "content": (
                    "Develop three creative directions based on this "
                    "original idea. Preserve its central concept and "
                    "apply balanced creativity:\n\n"
                    + idea
                ),
            },
        ],
        temperature=0.7,
    )

def is_done_response(response):
    cleaned = response.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)
    cleaned = cleaned.strip().strip("*` ").upper()
    return cleaned == "DONE"


def parse_guided_response(response):
    """Parse QUESTION plus three numbered examples, tolerating common formats."""
    cleaned = response.strip()
    cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)

    question = ""
    examples = []
    reading_examples = False

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()

        question_match = re.match(r"(?i)^question\s*:\s*(.+)$", line)
        if question_match:
            question = question_match.group(1).strip()
            continue

        if re.match(r"(?i)^example\s+answers\s*:", line):
            reading_examples = True
            continue

        if reading_examples:
            example_match = re.match(r"^\s*(?:[-*]\s*)?([1-3])[.)]\s*(.+)$", line)
            if example_match:
                examples.append(example_match.group(2).strip())

    # Remove accidental duplicate examples while preserving order.
    unique_examples = []
    for example in examples:
        if example and example not in unique_examples:
            unique_examples.append(example)

    return question, unique_examples[:3]


def guided_director(idea, creation_type, previous_answers):
    instructions = """
You are an expert AI Prompt Director conducting a short creative briefing.

Your job is to determine whether the user's idea genuinely needs
clarification before creating a high-quality generation prompt.

IMPORTANT: Your default decision should be DONE, not another question.

STRICT STOPPING RULES:

- Ask ZERO to FOUR questions total.
- FOUR is an absolute maximum, NEVER a target.
- Most ideas should need zero, one, or two questions.
- A detailed idea may need NO questions at all.
- After every user answer, reassess whether more information is
  genuinely essential.
- If enough information is available, immediately return DONE.
- After four questions have been answered, return DONE.

WHEN TO ASK A QUESTION:

Only ask if all three conditions are true:
1. An important part of the user's intended result is missing.
2. The missing information cannot reasonably be inferred.
3. Different possible answers would substantially change the
   final image or video prompt.

If a detail would merely make the prompt more specific or polished,
DO NOT ask about it. Use your creative judgment instead.

WHEN TO RETURN DONE:

Return DONE when the main subject or concept and its setting or
situation are clear enough to create a strong prompt.

For IMAGE prompts, you can normally decide these details yourself:
- Composition and camera perspective.
- Lighting and color palette.
- Mood and atmosphere.
- Background and foreground details.
- Artistic treatment and visual refinement.

For VIDEO prompts, you can normally decide these details yourself:
- Camera movement.
- Pacing and progression.
- Lighting and atmosphere.
- Cinematic treatment.
- Supporting environmental details.

DO NOT ask unnecessary questions about:
- Exact time of day or weather intensity.
- Additional landmarks or recognizable buildings.
- Specific color palettes.
- Extra foreground or background objects.
- Mood or atmosphere when a reasonable choice can be made.
- Camera angle or perspective when a suitable choice can be made.
- Decorative or minor visual details.

Choose suitable details yourself unless they are central to
the user's intended concept.

EXAMPLES OF THE CORRECT DECISION:

Example 1:
The user provides a detailed idea specifying the subject, location,
weather, lighting, style, and composition.
Correct response: DONE

Example 2:
The user says: "A futuristic underwater city."
You may ask ONE important question about its defining visual
direction or concept. After receiving the answer, reassess and
normally finish rather than asking about optional details.

Example 3:
The user says: "A dragon flying over a medieval castle."
You may ask ONE question if an important creative direction is
unclear. Do not continue asking about optional colors, lighting,
camera angles, or background details.

ADDITIONAL RULES:

- Ask only ONE question at a time.
- Never repeat a question.
- Never ask for information already provided.
- Every question must have exactly THREE complete example answers.
- Do not ask questions just to reach four.
- Do not include introductions, explanations, or commentary.
- Do not generate the final prompt yourself. Return DONE when ready.

IF ASKING A QUESTION, USE EXACTLY THIS FORMAT:

QUESTION: [One relevant question]

EXAMPLE ANSWERS:
1. [First complete example answer]
2. [Second complete example answer]
3. [Third complete example answer]

IF NO FURTHER QUESTION IS GENUINELY NEEDED, RETURN ONLY:

DONE
"""

    history = "\n\n".join(previous_answers) if previous_answers else "No answers yet."

    return ask_ollama(
        [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    f"Creation type: {creation_type}\n"
                    f"Original idea: {idea}\n"
                    f"Questions already asked and answers received:\n{history}\n\n"
                    "Decide whether another question is genuinely essential. "
                    "Remember: DONE is the default. Do not ask about optional "
                    "creative preferences that you can decide yourself."
                ),
            },
        ],
        temperature=0.1,
    )

def generate_guided_final_prompt(idea, creation_type, previous_answers):
    answer_history = "\n\n".join(previous_answers) if previous_answers else "No additional answers."

    return ask_ollama(
        [
            {"role": "system", "content": prompt_instructions(creation_type)},
            {
                "role": "user",
                "content": (
                    f"Original idea:\n{idea}\n\n"
                    f"User's answers to the creative questions:\n{answer_history}\n\n"
                    "Combine the original idea and all relevant answers into "
                    "one coherent, finished generation prompt. Do not omit "
                    "important details from the answers."
                ),
            },
        ],
        temperature=0.7,
    )


def initialize_state():
    defaults = {
        "guided_status": "idle",
        "guided_answers": [],
        "guided_question_count": 0,
        "guided_question": "",
        "guided_examples": [],
        "guided_idea": "",
        "guided_creation_type": "Image",
        "guided_final_prompt": "",
        "guided_error": "",
        "guided_raw_response": "",
        "generated_prompt": "",
        "creative_result": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def accept_guided_response(response):
    """Store a valid next question, or generate the final prompt on DONE."""
    st.session_state["guided_raw_response"] = response
    st.session_state["guided_error"] = ""

    if is_done_response(response):
        with st.spinner("Qwen is creating your final prompt..."):
            final_prompt = generate_guided_final_prompt(
                st.session_state["guided_idea"],
                st.session_state["guided_creation_type"],
                st.session_state["guided_answers"],
            )
        st.session_state["guided_final_prompt"] = final_prompt
        st.session_state["guided_status"] = "complete"
        st.session_state["guided_question"] = ""
        st.session_state["guided_examples"] = []
        return

    question, examples = parse_guided_response(response)

    if not question or len(examples) != 3:
        st.session_state["guided_status"] = "error"
        st.session_state["guided_error"] = (
            "Qwen's response did not contain a question and exactly three "
            "recognizable example answers. Your existing answers have been preserved."
        )
        return

    st.session_state["guided_question"] = question
    st.session_state["guided_examples"] = examples
    st.session_state["guided_question_count"] += 1
    st.session_state["guided_status"] = "asking"


def start_guided_session(idea, creation_type):
    st.session_state["guided_answers"] = []
    st.session_state["guided_question_count"] = 0
    st.session_state["guided_question"] = ""
    st.session_state["guided_examples"] = []
    st.session_state["guided_final_prompt"] = ""
    st.session_state["guided_error"] = ""
    st.session_state["guided_raw_response"] = ""
    st.session_state["guided_idea"] = idea.strip()
    st.session_state["guided_creation_type"] = creation_type
    st.session_state["guided_status"] = "asking"

    response = guided_director(
        st.session_state["guided_idea"],
        st.session_state["guided_creation_type"],
        st.session_state["guided_answers"],
    )
    accept_guided_response(response)


def submit_guided_answer(selected_answer):
    question = st.session_state["guided_question"]
    st.session_state["guided_answers"].append(
        f"Question: {question}\nAnswer: {selected_answer}"
    )

    # Four questions is a hard maximum. Generate the final prompt now.
    if st.session_state["guided_question_count"] >= 4:
        with st.spinner("Qwen is creating your final prompt..."):
            st.session_state["guided_final_prompt"] = generate_guided_final_prompt(
                st.session_state["guided_idea"],
                st.session_state["guided_creation_type"],
                st.session_state["guided_answers"],
            )
        st.session_state["guided_status"] = "complete"
        st.session_state["guided_question"] = ""
        st.session_state["guided_examples"] = []
        return

    # Otherwise let Qwen decide whether another question is necessary.
    with st.spinner("Qwen is considering your answer..."):
        response = guided_director(
            st.session_state["guided_idea"],
            st.session_state["guided_creation_type"],
            st.session_state["guided_answers"],
        )
        accept_guided_response(response)


def reset_app():
    st.session_state["creation_type"] = "Image"
    st.session_state["mode"] = "Quick Enhance"
    st.session_state["idea_input"] = ""

    keys_to_reset = (
        "guided_status",
        "guided_answers",
        "guided_question_count",
        "guided_question",
        "guided_examples",
        "guided_idea",
        "guided_creation_type",
        "guided_final_prompt",
        "guided_error",
        "guided_raw_response",
        "generated_prompt",
        "creative_result",
    )
    for key in keys_to_reset:
        st.session_state.pop(key, None)


# -------------------- APP INTERFACE --------------------

st.set_page_config(
    page_title="AI Prompt Director",
    page_icon="🎬",
    layout="centered",
)

initialize_state()

st.title("🎬 AI Prompt Director")
st.write(
    "Transform your creative ideas into detailed, "
    "high-quality AI image and video prompts."
)
st.divider()

st.subheader("1. What would you like to create?")
creation_type = st.radio(
    "Select your creation type:",
    ["Image", "Video"],
    horizontal=True,
    key="creation_type",
)

st.divider()
st.subheader("2. Choose your working mode")
mode = st.radio(
    "How would you like AI Prompt Director to help?",
    ["Quick Enhance", "Guided Director", "Creative Exploration"],
    help="Choose how you want to develop your idea.",
    key="mode",
)

if mode == "Quick Enhance":
    st.caption("Improve your idea immediately using sensible creative defaults.")
elif mode == "Guided Director":
    st.caption("Answer up to four relevant questions. Qwen can finish earlier when it has enough information.")
else:
    st.caption("Explore three distinct creative directions for your idea.")

st.divider()
st.subheader("3. Describe your idea")
idea = st.text_area(
    "What would you like to create?",
    placeholder="Example: A futuristic underwater city illuminated by glowing marine life...",
    height=150,
    key="idea_input",
)
st.caption(f"Creation type: {creation_type} | Mode: {mode}")


# -------------------- QUICK ENHANCE --------------------

if mode == "Quick Enhance":
    if st.button("✨ Enhance My Prompt", type="primary", use_container_width=True):
        if not idea.strip():
            st.warning("Please describe your idea first.")
        else:
            try:
                with st.spinner("Qwen 3.5 9B is enhancing your prompt..."):
                    st.session_state["generated_prompt"] = quick_enhance(
                        idea.strip(), creation_type
                    )
            except requests.exceptions.RequestException as error:
                st.error(f"Unable to connect to Ollama: {error}")

    if st.session_state["generated_prompt"]:
        st.subheader("Your Improved Prompt")
        st.text_area(
            "Copy your finished prompt below:",
            value=st.session_state["generated_prompt"],
            height=250,
            key="quick_prompt_output",
        )


# -------------------- GUIDED DIRECTOR --------------------

if mode == "Guided Director":
    if st.session_state["guided_status"] in ("idle", "complete"):
        if st.button(
            "🎯 Start Guided Director" if st.session_state["guided_status"] == "idle"
            else "🔄 Start a New Guided Session",
            type="primary",
            use_container_width=True,
        ):
            if not idea.strip():
                st.warning("Please describe your idea first.")
            else:
                try:
                    with st.spinner("Qwen 3.5 9B is analyzing your idea..."):
                        start_guided_session(idea, creation_type)
                except requests.exceptions.RequestException as error:
                    st.session_state["guided_status"] = "error"
                    st.session_state["guided_error"] = f"Unable to connect to Ollama: {error}"

    if st.session_state["guided_status"] == "asking":
        st.subheader(
            f"Guided Director — Question "
            f"{st.session_state['guided_question_count']} of 4 maximum"
        )
        st.write(f"**{st.session_state['guided_question']}**")

        form_key = f"guided_answer_form_{st.session_state['guided_question_count']}"
        with st.form(key=form_key):
            choice = st.radio(
                "Choose an example answer, or write your own:",
                st.session_state["guided_examples"] + ["Write my own answer"],
                key=f"guided_answer_choice_{st.session_state['guided_question_count']}",
            )
            custom_answer = st.text_area(
                "Your own answer (only needed if selected above):",
                placeholder="Enter your answer here...",
                height=100,
                key=f"guided_custom_answer_{st.session_state['guided_question_count']}",
            )
            submitted = st.form_submit_button("Submit Answer", type="primary")

        if submitted:
            if choice == "Write my own answer":
                selected_answer = custom_answer.strip()
            else:
                selected_answer = choice.strip()

            if not selected_answer:
                st.warning("Please enter your answer first.")
            else:
                try:
                    submit_guided_answer(selected_answer)
                    st.rerun()
                except requests.exceptions.RequestException as error:
                    st.session_state["guided_status"] = "error"
                    st.session_state["guided_error"] = f"Unable to connect to Ollama: {error}"

    elif st.session_state["guided_status"] == "complete":
        st.success(
            f"Your final {st.session_state['guided_creation_type'].lower()} "
            "prompt is ready."
        )
        st.subheader("Your Final Prompt")
        st.text_area(
            "Copy your finished prompt below:",
            value=st.session_state["guided_final_prompt"],
            height=300,
            key="guided_final_prompt_output",
        )
        st.caption(
            f"Questions answered: {len(st.session_state['guided_answers'])} "
            f"(maximum 4)."
        )

    elif st.session_state["guided_status"] == "error":
        st.error(st.session_state["guided_error"])
        if st.session_state["guided_raw_response"]:
            with st.expander("View Qwen's raw response"):
                st.code(st.session_state["guided_raw_response"])
        if st.button("Retry Guided Director", type="primary"):
            try:
                with st.spinner("Qwen is trying again..."):
                    response = guided_director(
                        st.session_state["guided_idea"],
                        st.session_state["guided_creation_type"],
                        st.session_state["guided_answers"],
                    )
                    accept_guided_response(response)
                st.rerun()
            except requests.exceptions.RequestException as error:
                st.error(f"Unable to connect to Ollama: {error}")


# -------------------- CREATIVE EXPLORATION --------------------

if mode == "Creative Exploration":
    if st.button(
        "🎨 Explore Creative Directions",
        type="primary",
        use_container_width=True,
    ):
        if not idea.strip():
            st.warning("Please describe your idea first.")
        else:
            try:
                with st.spinner("Qwen 3.5 9B is exploring creative directions..."):
                    st.session_state["creative_result"] = creative_exploration(
                        idea.strip(), creation_type
                    )
            except requests.exceptions.RequestException as error:
                st.error(f"Unable to connect to Ollama: {error}")

    if st.session_state["creative_result"]:
        st.subheader("Three Creative Directions")
        st.text_area(
            "Your creative directions:",
            value=st.session_state["creative_result"],
            height=400,
            key="creative_directions_output",
        )


st.divider()
st.button(
    "🔄 Start Over",
    use_container_width=True,
    on_click=reset_app,
)
