import re
import os
import json
import time as _time

# ── HuggingFace housekeeping ─────────────────────────────────────────────────
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
# ─────────────────────────────────────────────────────────────────────────────

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Negative prompt builder ───────────────────────────────────────────────────

_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, floating limbs, disconnected body parts, "
    "overexposed, underexposed, grainy, noise, moire pattern, shimmer"
)

_NEG_INDOOR        = "harsh outdoor lighting, direct sunlight"
_NEG_OUTDOOR       = "studio background, indoor lighting"
_NEG_EXPLICIT      = "censored, mosaic, pixelated, black bar, blurred genitals"
_NEG_PORTRAIT_SHOT = "wide angle distortion, fish eye, full body shot"
_NEG_WIDE          = "close-up, portrait crop, tight frame"
_NEG_NIGHT         = "overexposed, bright daylight, blown highlights"
_NEG_DAY           = "underexposed, dark shadows, black crush"
_NEG_MULTI         = "merged bodies, fused figures, incorrect number of people"
_NEG_PORTRAIT_ORI  = "landscape orientation, letterbox, pillarbox, horizontal crop, widescreen framing"
_NEG_VHS           = "clean digital, sharp edges, 4K, high resolution, pristine quality"
_NEG_HORROR        = "bright happy lighting, warm tones, cheerful atmosphere, soft light"
_NEG_FASHION       = "casual handheld, amateur footage, flat lighting, unposed"

_NEG_ANIME         = "photorealistic, live action, real person, CGI, 3D render, western cartoon, flat shading"
_NEG_2DCARTOON     = "photorealistic, 3D render, CGI, anime, live action, flat digital art, no line work"
_NEG_3DCGI         = "photorealistic, live action, 2D flat, hand-drawn, sketch, anime, watercolour"
_NEG_STOPMOTION    = "smooth motion, CGI, photorealistic, digital, fluid movement, motion blur"
_NEG_COMICBOOK     = "photorealistic, soft gradients, 3D render, painterly, no line art, anime"
_NEG_CELSHADED     = "photorealistic, soft shading, gradients, painterly, hand-drawn lines, anime"
_NEG_ROTOSCOPE     = "fully animated, cartoon, CGI, no live action base, unnatural movement"
_NEG_CYBERPUNK     = "natural lighting, pastoral, warm tones, daylight, photorealistic skin, muted colour"
_NEG_SCIFI         = "medieval, fantasy, nature, pastoral, historical, period costume, warm earthy tones"

def _build_negative_prompt(result: str, user_input: str, is_portrait: bool = False, style_preset: str = "") -> str:
    combined = (result + " " + user_input + " " + style_preset).lower()
    extras = []

    if any(w in combined for w in ["indoor", "room", "interior", "bedroom", "kitchen", "office"]):
        extras.append(_NEG_OUTDOOR)
    elif any(w in combined for w in ["outdoor", "street", "beach", "forest", "park", "exterior"]):
        extras.append(_NEG_INDOOR)

    if any(w in combined for w in ["pussy", "cock", "penis", "vagina", "nude", "naked", "explicit", "nipple", "breast"]):
        extras.append(_NEG_EXPLICIT)

    if any(w in combined for w in ["close-up", "close up", "face shot", "headshot"]):
        extras.append(_NEG_PORTRAIT_SHOT)
    elif any(w in combined for w in ["wide shot", "wide angle", "aerial", "bird's-eye", "establishing"]):
        extras.append(_NEG_WIDE)

    if any(w in combined for w in ["night", "dark", "moonlight", "dimly lit", "candlelight"]):
        extras.append(_NEG_NIGHT)
    elif any(w in combined for w in ["daylight", "sunny", "golden hour", "bright", "midday"]):
        extras.append(_NEG_DAY)

    if any(w in combined for w in ["two women", "two men", "two people", "both", "together", "couple", "they "]):
        extras.append(_NEG_MULTI)

    if is_portrait or "portrait vertical" in style_preset.lower() or "9:16" in style_preset:
        extras.append(_NEG_PORTRAIT_ORI)

    if "lo-fi" in style_preset.lower() or "vhs" in style_preset.lower():
        extras.append(_NEG_VHS)
    if "horror" in style_preset.lower():
        extras.append(_NEG_HORROR)
    if "fashion editorial" in style_preset.lower():
        extras.append(_NEG_FASHION)


    # Animation styles
    if "anime" in style_preset.lower():
        extras.append(_NEG_ANIME)
    if "2d cartoon" in style_preset.lower():
        extras.append(_NEG_2DCARTOON)
    if "3d cgi" in style_preset.lower():
        extras.append(_NEG_3DCGI)
    if "stop motion" in style_preset.lower():
        extras.append(_NEG_STOPMOTION)
    if "comic book" in style_preset.lower():
        extras.append(_NEG_COMICBOOK)
    if "cel-shaded" in style_preset.lower():
        extras.append(_NEG_CELSHADED)
    if "rotoscope" in style_preset.lower():
        extras.append(_NEG_ROTOSCOPE)
    if "cyberpunk" in style_preset.lower():
        extras.append(_NEG_CYBERPUNK)
    if "sci-fi" in style_preset.lower():
        extras.append(_NEG_SCIFI)

    parts = [_NEG_BASE] + extras
    return ", ".join(parts)


class LTX2PromptArchitect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bypass": ("BOOLEAN", {"default": False, "tooltip": "When ON, skips the LLM entirely and sends your text straight to the prompt encoder. Use for manual prompts or testing."}),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "a woman walks through a rain-soaked city street at night",
                    "tooltip": "Describe what you want to happen. Can be a rough idea, a sentence, or numbered steps (1. she stands 2. she walks). The LLM expands this into a full cinematic prompt."
                }),
                "creativity": ([
                    "0.5 - Strict & Literal",
                    "0.8 - Balanced Professional",
                    "1.0 - Artistic Expansion"
                ], {"default": "0.8 - Balanced Professional", "tooltip": "Controls how closely the LLM sticks to your input. 0.5 is very literal and precise — closest to your exact words. 0.8 is balanced with professional cinematic language. 1.0 adds more creative flair and expansion beyond your input."}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Set a fixed seed to get the same prompt expansion every run. Use -1 for a random result each time."
                }),
                "invent_dialogue": ("BOOLEAN", {"default": True, "tooltip": "When ON, the LLM invents natural spoken dialogue for characters woven into the scene. When OFF, only uses dialogue you wrote yourself (in quotes), or generates no dialogue at all."}),
                "keep_model_loaded": ("BOOLEAN", {"default": False, "tooltip": "Keep the LLM in VRAM between runs for faster generation. Turn OFF to free VRAM immediately after each run — recommended if you have less than 16GB VRAM."}),
                "offline_mode": ("BOOLEAN", {"default": False, "tooltip": "Turn ON if you have no internet. Uses locally cached models only. Turn OFF to allow auto-download from HuggingFace on first run."}),
                "frame_count": ("INT", {
                    "default": 192,
                    "min": 24,
                    "max": 960,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Match this to your video LENGTH setting. Controls pacing — the LLM uses this to calculate how many actions fit in the clip. 24fps = 1 second, so 192 = 8 seconds."
                }),
                "style_preset": (list(LTX2PromptArchitect.STYLE_PRESETS.keys()), {
                    "default": "None — let the LLM decide",
                    "tooltip": "Sets the visual aesthetic for the prompt — lighting, colour, camera, mood. Also drives the FPS output pin automatically: cinematic presets = 24, realistic/action = 30. Wire FPS to your video and audio save nodes."
                }),
                "portrait_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force 9:16 vertical framing for TikTok, Reels, and Shorts. LTX-2.3 has native portrait support — use this to take advantage of it. Overrides style preset orientation."
                }),
                # ── Model selector ──────────────────────────────────────────
                "model": ([
                    "8B - NeuralDaredevil (High Quality)",
                    "3B - Llama-3.2 Abliterated (Low VRAM)",
                ], {"default": "8B - NeuralDaredevil (High Quality)", "tooltip": "Choose your LLM. 8B gives better quality prompts and handles explicit content well. 3B is faster and uses less VRAM. Both download automatically on first run."}),
                # ── Local paths for offline mode ────────────────────────────
                "local_path_8b": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g. C:\\Users\\YOU\\.cache\\huggingface\\hub\\models--mlabonne--NeuralDaredevil-8B-abliterated\\snapshots\\YOUR_HASH",
                    "tooltip": "Optional. Paste the full path to your locally downloaded NeuralDaredevil 8B snapshot folder. Leave blank to use the HuggingFace cache automatically."
                }),
                "local_path_3b": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Local path to Llama-3.2 3B snapshot folder",
                    "tooltip": "Optional. Paste the full path to your locally downloaded Llama 3.2 3B snapshot folder. Leave blank to use the HuggingFace cache automatically."
                }),
            },
            "optional": {
                "scene_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: vision description from LTX-2 Vision Describe node",
                    "tooltip": "Wire the output from the LTX-2 Vision Describe node here. The LLM will use your image as the authoritative starting point and animate it forward from your prompt."
                }),
                "lora_triggers": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: LoRA trigger words e.g. 'ohwx woman, film grain'",
                    "tooltip": "Paste your LoRA trigger words here. They will be injected at the very start of every generated prompt automatically — never buried or forgotten."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("PROMPT", "PREVIEW", "NEG_PROMPT", "FPS", "HISTORY")
    FUNCTION = "generate"
    CATEGORY = "LTX2"

    # ── Style presets ─────────────────────────────────────────────────────────
    # Maps dropdown label → (style instruction, portrait flag)
    STYLE_PRESETS = {
        "None — let the LLM decide": ("", False),
        # Cinematic / Narrative
        "Slow-burn thriller": (
            "STYLE: Slow-burn psychological thriller. Tight framing, long held shots, shallow depth of field. "
            "Colour palette: desaturated teal and amber. Sound design is sparse — silence punctuated by single sounds. "
            "Camera moves deliberately and slowly. Tension built through restraint, not action.", False),
        "Handheld documentary": (
            "STYLE: Handheld documentary. Camera moves with the subject, never static. Slight shake on movement. "
            "Natural available light only — no studio lighting. Colour grade: flat, slightly washed. "
            "Intimate and observational — camera follows, never leads.", False),
        "High fashion editorial": (
            "STYLE: High fashion editorial. Striking, composed frames. Hard directional lighting with deep shadows. "
            "Colour palette: high contrast, often monochrome or single accent colour. "
            "Movement is deliberate and posed — model-aware. Camera movements are slow and precise. "
            "ENVIRONMENT NOTE: Do not invent luxury props, chandeliers, marble, or opulent settings "
            "unless the user described them. Apply the editorial aesthetic to whatever location the user specified.", False),
        "Noir — deep shadows, venetian light": (
            "STYLE: Classic noir. Low-key lighting, venetian blind shadow patterns across faces and walls. "
            "Black and white or heavily desaturated with single colour accent. "
            "Camera angles: low, Dutch tilt, shot through objects. Mood is foreboding and fatalistic.", False),
        "Golden hour drama": (
            "STYLE: Golden hour drama. Warm amber and orange light from a low sun. Heavy lens flare. "
            "Soft shadows, glowing skin tones. Wide lenses. Emotional, sweeping camera movement. "
            "Colour grade: warm, slightly overexposed highlights.", False),
        "Horror — desaturated, harsh contrast": (
            "STYLE: Horror. Heavily desaturated colour, crushed blacks. Harsh top-down or under-lighting. "
            "Camera movements are slow and uneasy — never reassuring. "
            "Framing leaves negative space — empty doorways, dark corners. No warmth in the image.", False),
        # Erotic / Adult
        "Erotic cinema — tasteful, cinematic": (
            "STYLE: Tasteful erotic cinema. Warm, intimate lighting — practical sources only. "
            "Shallow depth of field. Camera moves slowly and deliberately. "
            "Colour grade: warm skin tones, soft highlights. "
            "Sensual but not pornographic — implication over explicit detail. Slow, breathing pace. "
            "SCOPE NOTE: This style sets the aesthetic and mood only. "
            "It does NOT grant permission to add nudity, undressing, or intimate acts beyond what the user described. "
            "Describe only what was asked for — the style wraps it, it does not expand it.", False),
        "Explicit — direct, anatomical": (
            "STYLE: Explicit adult content. Direct lighting — bodies clearly lit with no flattering shadow. "
            "Camera is close and functional — shows exactly what is happening without cinematic softening. "
            "No romantic framing. Blunt and specific. Anatomical language used directly. "
            "IMPORTANT: This style affects aesthetic and language only — it does NOT expand the scope of the scene beyond what the user asked for. "
            "Describe only what the user requested. Do not add acts, exposure, or nudity the user did not write.", False),
        "Voyeur — handheld, observational": (
            "STYLE: Voyeuristic. The camera is a person — someone who found this moment and is trying not to be noticed. "
            "CAMERA BEHAVIOUR — MANDATORY: "
            "Unless the user explicitly said 'static', the camera is ALWAYS in motion. "
            "It bobs and drifts with the natural sway of someone walking or standing. "
            "The motion is involuntary — slight vertical bounce, gentle lateral drift, micro-rotations. "
            "The camera NEVER repositions to get a better angle. It stays at the height and position of the person holding it — "
            "hip height if they are trying to be discreet, chest height if partially hidden, never raised to eye level for a clean shot. "
            "FORBIDDEN camera moves: crane up, dolly in, rack focus, orbit, push in, pull back, pan to follow. "
            "ALLOWED camera behaviour: drifts, bobs, tilts slightly as the subject moves, briefly obscured by a passing person or shelf, "
            "loses the subject for a frame and finds them again. "
            "The framing is imperfect — the subject may be partially cut off, slightly out of focus at the edges, "
            "or briefly blocked. This is what makes it feel real. "
            "Natural available light only — no fill, no flash, no colour grading. "
            "The subject is unaware. The camera does not announce itself. "
            "CRITICAL: The subject's actions are exactly as the user described — do not invent, reverse, or reframe them. "
            "If the user said she is getting dressed, she is getting dressed. If the user said she is undressing, she is undressing. "
            "The camera observes what is happening — it does not change what is happening.", False),
        "Softcore editorial — lingerie-adjacent": (
            "STYLE: Softcore editorial. Fashion-magazine aesthetic. Clean, even lighting. "
            "Colour grade: warm neutrals and soft pastels. "
            "Camera is composed — lingerie-level sensuality, no explicit content. Movement is slow and posed. "
            "SCOPE NOTE: This style sets the aesthetic only. "
            "Do NOT add undressing, nudity, or intimate acts the user did not ask for. "
            "If the user described someone sitting or standing clothed, they stay clothed. "
            "The style applies to framing and mood — not to what happens in the scene.", False),
        "Amateur — naturalistic, raw": (
            "STYLE: Amateur home video aesthetic. Slightly overexposed. Natural indoor lighting — lamps, overhead. "
            "Camera is handheld and slightly uncertain. No cinematic framing. "
            "Colour: ungraded, as-shot. The imperfection is intentional.", False),
        # Action / Energy
        "Action blockbuster": (
            "STYLE: Action blockbuster. Fast kinetic energy. Dutch angles, crash zooms, whip pans. "
            "Colour grade: teal and orange, high contrast. "
            "Camera is never still — it moves with every impact. Slow motion inserts on key moments.", False),
        "Sports documentary": (
            "STYLE: Sports documentary. Tracking shots following the athlete. Telephoto compression. "
            "Slow motion bursts at peak moments. Natural sound — crowd noise, impact, breathing. "
            "Colour grade: clean and neutral. Camera is athletic — it moves like it is competing too.", False),
        "Music video — stylised": (
            "STYLE: Music video. Rhythm-cut visual language — movement is driven by the beat. "
            "High contrast colour grade with stylised palette. "
            "Mix of tight close-ups and dramatic wide shots. Camera movement is expressive, not documentary. "
            "AUDIO: Music is present — describe the track's energy, tempo, and texture as physical sound: "
            "'a driving four-on-the-floor kick', 'sharp hi-hats', 'a warm bass line pulsing beneath the mix'. "
            "Sync camera and body movement to the implied beat.", False),
        # Aesthetic / Visual
        "Lo-fi home video — VHS": (
            "STYLE: Lo-fi home video. VHS tape aesthetic — slightly washed colour, faint scan lines, soft edges. "
            "Colour grade: faded, slightly green-shifted. Camera is handheld and casual. "
            "Intimate domestic setting implied. Imperfection is the aesthetic. "
            "IMPORTANT: This style describes HOW the scene is shot — not what is in it. "
            "All people, subjects, and actions described by the user must still appear in the scene. "
            "Do not replace the user's scene with an empty room, leftover objects, or nostalgic cutaways. "
            "Film the scene the user described, through a VHS camera.", False),
        "Hyper-real 4K — clinical sharpness": (
            "STYLE: Hyper-real 4K. Clinical sharpness — every texture, pore, and fibre rendered in full detail. "
            "Even lighting, no blown highlights, no crushed blacks. "
            "Camera movement is minimal and precise. The image is almost uncomfortably detailed.", False),
        "Dreamy — soft focus, slow motion": (
            "STYLE: Dreamy aesthetic. Soft focus edges with sharp centre. Pastel colour bleed. "
            "Movement is slow — the frame breathes rather than cuts. "
            "Lens: wide aperture with heavy bokeh. Light sources bloom and halo.", False),
        "Gritty realism — flat, natural light": (
            "STYLE: Gritty realism. Flat colour grade, no cinematic enhancement. Natural light only — "
            "whatever is available in the location. Camera is direct and unsentimental. "
            "No stylisation. The scene is shot as if it is actually happening.", False),
        # Speciality
        "POV — first person, immersive": (
            "STYLE: First-person POV. The camera IS the viewer's eyes. "
            "Frame moves as a head would — natural breathing movement, slight tilt on turns. "
            "Everything is seen, not watched. Close physical detail — hands, surfaces, faces at speaking distance.", False),
        "Portrait vertical — 9:16 mobile": (
            "STYLE: Native portrait video, 9:16 aspect ratio. Optimised for mobile — TikTok, Reels, Shorts. "
            "Frame is vertical throughout. Tight head-to-torso framing. "
            "Action moves vertically in frame. Camera stays close. No wide horizontal composition.", True),
        # Animation
        "Anime — Japanese animation": (
            "STYLE: Japanese anime. Hand-drawn animation aesthetic — clean ink outlines, flat colour fills with "
            "subtle cel shading. Large expressive eyes, stylised facial features. "
            "Colour palette: vivid, high saturation with strong accent colours. "
            "Motion: fluid on key poses, held on reaction shots — classic anime timing with smear frames on fast movement. "
            "Background art is painterly and detailed behind simpler foreground characters. "
            "Camera: dynamic angles, speed lines on action, slow drift on emotional beats. "
            "Render every subject — human, animal, object — in this style regardless of what was described.", False),
        "2D cartoon — hand-drawn": (
            "STYLE: Classic hand-drawn 2D cartoon. Expressive ink outlines with variable line weight — thick on silhouette, thin on interior detail. "
            "Flat colour fills, minimal shading, bold colour palette. "
            "Movement uses squash-and-stretch — characters exaggerate physics for comedic or emotive effect. "
            "Timing is snappy — fast actions are faster than real life, held poses linger longer. "
            "Background art is simplified and stylised, never photorealistic. "
            "Camera: mostly static or slow panning, occasional dramatic zoom. "
            "Render every subject in this style regardless of what was described.", False),
        "3D CGI — Pixar/DreamWorks": (
            "STYLE: High-end 3D CGI animation in the style of Pixar or DreamWorks. "
            "Subsurface scattering on skin and organic surfaces — warmth and translucency visible in light. "
            "Highly detailed surface textures: pores, fur, feathers, fabric weave all rendered at full resolution. "
            "Expressive faces with large eyes capable of subtle micro-expressions. "
            "Warm, soft three-point lighting with dappled environmental light and gentle shadows. "
            "Camera: smooth cinematic moves — slow push-ins, gentle orbits, rack focus between characters. "
            "Colour grade: warm, slightly saturated, storybook palette. "
            "Render every subject in this style regardless of what was described.", False),
        "Stop motion — claymation": (
            "STYLE: Stop motion claymation. Physical clay or puppet aesthetic — visible fingerprints and tool marks in surfaces, "
            "slight imperfections in every frame that reveal the handmade origin. "
            "Movement is slightly jerky and deliberate — 12 frames per second gives it weight and tactility. "
            "Textures: matte, tactile, slightly waxy. Colours are saturated but not digital. "
            "Sets are physical miniatures — tangible depth, real shadows from practical lights. "
            "Camera: locked off or on simple mechanical rigs — no digital smoothing. "
            "Render every subject in this style regardless of what was described.", False),
        "Comic book / graphic novel": (
            "STYLE: Comic book or graphic novel. Bold ink outlines, halftone dot patterns in shadow areas. "
            "Colour is flat with hard-edged shadows — no soft gradients. "
            "Panel energy: dynamic Dutch angles, strong perspective distortion on action, tight close-ups on emotion. "
            "Speed lines radiate from points of impact or fast movement. "
            "Colour palette: high contrast, often limited to 3-5 colours per scene with heavy black ink. "
            "Camera moves like a comic panel transition — hard cuts between angles, no smooth motion blur. "
            "Render every subject in this style regardless of what was described.", False),
        "Cel-shaded — flat colour 3D": (
            "STYLE: Cel-shaded 3D. Three-dimensional geometry rendered with flat, stepped colour fills — no soft gradients. "
            "Hard shadow threshold: shadow areas are a single flat darker tone, lit areas a single flat lighter tone. "
            "Ink outlines on all silhouettes and major edges. "
            "The image reads as animated despite being 3D — the shading removes photorealism entirely. "
            "Colour palette: clean, bold, graphic. "
            "Camera: precise and composed — treats 3D space like a 2D stage. "
            "Render every subject in this style regardless of what was described.", False),
        "Rotoscope — animated over live action": (
            "STYLE: Rotoscoped animation. The movement is real — traced from live action footage — "
            "giving it uncanny physical accuracy within a hand-drawn or painted surface. "
            "Outlines are hand-drawn over every frame: slightly wobbly, varying in weight, never perfectly clean. "
            "Colour is either painted in loose washes or held as flat fills inside the traced lines. "
            "The result feels simultaneously real and unreal — human movement with an illustrated skin. "
            "Background may be live action or painted. Camera movement follows the original footage exactly. "
            "Render every subject in this style regardless of what was described.", False),
        "Cyberpunk neon illustrated": (
            "STYLE: Cyberpunk illustrated. Neon-lit urban environment — magenta, cyan, electric blue, acid green. "
            "Hard rim lighting from neon signs carves subjects out of near-total darkness. "
            "Rain-slick surfaces reflect light in pools and streaks. "
            "The aesthetic blends hyper-detailed digital illustration with cinematic composition — "
            "not photorealistic, but not flat cartoon either. Think graphic novel meets blade runner. "
            "Typography and UI elements float in the environment as holographic overlays. "
            "Camera: low angles, wide lenses, dramatic fog and haze. "
            "Render every subject in this style regardless of what was described.", False),
        "Sci-fi — cinematic, practical": (
            "STYLE: Cinematic science fiction. Clean, practical-feeling environments — metal corridors, "
            "reinforced glass, industrial lighting rigs. Colour palette: cool blue-white with accent LEDs, "
            "deep shadow with hard point sources. No fantasy or magic — everything looks functional and built. "
            "Camera: wide establishing shots to sell the scale of the environment, then close on faces or hands "
            "for intimacy. Lens flare on light sources. Sound is mechanical — hum of systems, "
            "footsteps on metal grating, distant machinery. "
            "Render every subject in this style regardless of what was described.", False),
    }

    # ── FPS map — auto output based on style preset ───────────────────────────
    # Single INT output — wire to video save node and audio save node
    # 24 = cinematic  |  30 = realistic / action
    PRESET_FPS = {
        # Cinematic / Narrative
        "None — let the LLM decide":                24,
        "Slow-burn thriller":                       24,
        "Handheld documentary":                     30,  # TV/doc feel
        "High fashion editorial":                   24,
        "Noir — deep shadows, venetian light":      24,
        "Golden hour drama":                        24,
        "Horror — desaturated, harsh contrast":     24,
        # Adult / Sensual
        "Erotic cinema — tasteful, cinematic":      24,
        "Explicit — direct, anatomical":            30,
        "Voyeur — handheld, observational":         30,
        "Softcore editorial — lingerie-adjacent":   24,
        "Amateur — naturalistic, raw":              30,
        # Action / Energy
        "Action blockbuster":                       30,
        "Sports documentary":                       30,
        "Music video — stylised":                   30,
        # Aesthetic / Visual
        "Lo-fi home video — VHS":                   24,
        "Hyper-real 4K — clinical sharpness":       30,
        "Dreamy — soft focus, slow motion":         24,
        "Gritty realism — flat, natural light":     30,
        # Speciality
        "POV — first person, immersive":            30,
        "Portrait vertical — 9:16 mobile":          30,

        # Animation
        "Anime — Japanese animation":               24,
        "2D cartoon — hand-drawn":                  24,
        "3D CGI — Pixar/DreamWorks":                24,
        "Stop motion — claymation":                 24,
        "Comic book / graphic novel":               24,
        "Cel-shaded — flat colour 3D":              24,
        "Rotoscope — animated over live action":    24,
        "Cyberpunk neon illustrated":               30,
        "Sci-fi — cinematic, practical":            24,
    }

    # ── Model registry ────────────────────────────────────────────────────────
    MODELS = {
        "8B - NeuralDaredevil (High Quality)": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B - Llama-3.2 Abliterated (Low VRAM)": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
    }

    # ── System prompt ─────────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2.3, an AI video generation model. Your job is to expand a user's rough idea into a precise, director-level, video-ready prompt that extracts maximum quality from LTX-2.3's capabilities.

LTX-2.3 CAPABILITIES — use these fully:
- Handles complex prompts with multiple subjects, spatial relationships, layered actions, and stylistic constraints. Specificity wins — do not simplify.
- Rebuilt VAE renders fine detail: fabric weave, hair strands, surface texture, skin pores, material finish. Describe these explicitly.
- Stronger prompt adherence means you can direct camera movement alongside subject motion simultaneously.
- Native portrait support up to 1080x1920 — compose vertically when in portrait mode, not as cropped landscape.
- Improved audio vocoder — describe sound specifically: tone, intensity, environment, direction.
- Reduced motion freezing — static prompts still produce static output. Always include motion.

ANTI-HALLUCINATION RULE — this overrides everything else:
Only describe what the user asked for. Do NOT invent props, atmosphere, or mood elements the user did not mention.
Do NOT add: rose petals, candles, silk sheets, flowers, soft light, mist, rain, fog, smoke, butterflies, curtains blowing, glitter, sparkles, or any other atmospheric filler the user did not request.
Do NOT invent a location or setting. If the user gives only an action with no location, shoot it in a neutral unspecified space — do not conjure a warehouse, kitchen, forest, or any environment the user did not describe.
Do NOT invent abstract emotional sound — no "the heartbeat of the city", no "tension hums in the air", no musical overtones. Sound must be concrete and physical only.
Every detail must be either (a) directly from the user's input, (b) required by the active style preset, or (c) a necessary camera/lighting/staging decision to make the scene work visually.

PRIORITY ORDER — build the prompt in this sequence:
1. Video style & genre — use the STYLE INSTRUCTION as the aesthetic anchor. If none given, choose one that fits.
2. Camera orientation — if the subject should NOT be facing camera (e.g. "from behind", "follows her", "rear view", "over her shoulder"), state this as the VERY FIRST words. E.g. "Rear view." or "The camera follows her from behind." LTX defaults to front-facing — override it early and explicitly.
3. Camera angle & shot type — cinematographic terms: dolly, tracking shot, OTS, Dutch angle, bird's-eye, snorkel lens. Be specific.
4. Lens & optics — always state focal length and aperture: "85mm f/1.4", "24mm wide angle", "50mm macro". Controls edge sharpness and depth of field in LTX-2.3.
5. Character — age as a specific number always e.g. "a 31-year-old woman" — never omit. Then: hair texture (fine, coarse, wavy, tightly coiled), skin tone, body type, clothing with fabric and material detail (e.g. "a loose cotton t-shirt", "a black satin dress", "worn denim jeans with frayed hems"). Name body parts using the exact words the user used.
6. Scene & environment — location, time of day, lighting quality and direction, colour temperature, surface textures. Describe material and wear: "cracked concrete floor", "brushed steel countertop", "worn wooden floorboards". Only what the user described or logically necessary.
7. Spatial blocking — be explicit: left vs right, foreground vs background, distance between subjects, who faces what. Block it like a director. "She stands left of frame, back to camera. He sits on the right, facing her."

THEN — action and motion:
8. Action & motion — use VERBS. Specify: who moves, what moves, how they move, what the camera does — as four distinct things when relevant. "She turns her head and steps forward as the camera tracks right." Motion is driven by verbs. Do NOT write static, photo-like descriptions — if the user's input is inherently static, add environmental motion: wind moving hair, background figures walking, a flag rippling, leaves shifting. LTX-2.3 produces freeze frames from static prompts.
9. Texture & detail in motion — describe how materials behave: "the fabric pulls taut across her hips as she bends", "her hair lifts and separates in the wind", "the leather creases at the elbow as she reaches". LTX-2.3's VAE can render this — use it.
10. Camera movement — prose only, never bracketed. Not "(Pull back)" — "the shot pulls back to frame the empty corridor." Vocabulary: dolly in, rack focus, whip pan, push in, crane up, handheld drift, slow orbit, creep forward.
11. Audio — weave as short concrete clauses. Maximum 2 sounds active per beat. Describe tone and intensity: "a low metallic hum", "sharp heels on marble, each step crisp". NEVER abstract emotional audio — no "tension fills the air", no "atmosphere hums with dread". No [AMBIENT: ...] tags. MUSIC EXCEPTION: if the scene involves dancing, a club, a performance, or music is implied — describe the music as physical sound: tempo, bass weight, hi-hat rhythm, drop, swell. "A deep kick drum drives the tempo", "bass pulses through the floor". Music is sound — describe it concretely, do not silence it.
12. Dialogue — follow the DIALOGUE INSTRUCTION exactly. Inline prose with attribution and physical delivery. No [DIALOGUE: ...] tags.

UNDRESSING RULE — mandatory when clothing removal is implied or stated:
Dedicate a full narrative segment to undressing BEFORE any nudity or explicit act. Name each garment. Describe HOW it is removed step by step. Describe what is physically REVEALED at each step — include skin texture and how the fabric behaves as it moves. Do NOT jump from clothed to naked. Do NOT compress steps.

GARMENT CHOREOGRAPHY — use the correct physical sequence for each type:
- Shirt / t-shirt / crop top (full removal): fingers find and grip the hem at the waist → fabric gathered and pulled upward → shirt rises past the stomach → past the ribs → over the chest → pulled over the head and off the arms → discarded
- Shirt / t-shirt / crop top (lift only — not removed): fingers find the hem at the waist → grip the fabric → slowly gather and lift → fabric rises past the stomach → past the navel → past the ribs → chest comes into view → breasts fully exposed → held there. Every step is its own sentence. Do NOT compress into one line.
- Dress (pullover): hands grip the hem at the thighs → lifted up past the hips → past the waist → gathered over the chest → pulled over the head → falls away
- Dress (zip): hand reaches behind to find the zip → zip pulled slowly downward → fabric loosens and parts → dress slipped off the shoulders → slides down the body → falls to the floor
- Blouse / button-down: fingers work each button from top to bottom one at a time → fabric parts with each button → shrugged off the shoulders → slides down the arms → dropped
- Bra: hand reaches behind to the clasp → clasp unhooked → straps slacken → straps slipped off each shoulder in turn → cups fall away → removed and set aside
- Jeans / trousers: button popped → zip drawn down → waistband pushed down over the hips → fabric pushed down the thighs → stepped out of
- Underwear / knickers / thong: thumbs hooked into the waistband at the hips → pushed down → stepped out of

NO INVENTED RESOLUTION: Do NOT have the subject reverse, cover, or undo any action unless the user asked for it. If she lifts her shirt, it stays lifted. Do not write her pulling it back down or covering herself unless explicitly requested.

PORTRAIT MODE — 9:16 vertical: compose vertically from the start, not as cropped landscape. Tight head-to-torso framing. Action and camera movement flow vertically in frame. No wide horizontal compositions.

WRITING RULES:
- Present tense throughout
- Specificity wins — "a loose grey cotton t-shirt, slightly faded at the collar" beats "a shirt". LTX-2.3 can render the detail.
- Direct and concrete — "her red dress falls to the floor" beats "the crimson fabric cascades like a waterfall of desire"
- No vague filler: not "beautiful", "stunning", "gorgeous", "elegant" — describe what is actually visible on screen
- Layer complexity — LTX-2.3 holds structure under complex prompts. Use multiple actions, detailed environments, and camera direction together confidently.
- Flowing prose, not bullet lists

HARD OUTPUT RULES:
Output ONLY the prompt. No preamble. No "Sure!" or "Here's your prompt:". No checklist, compliance note, or summary at the end. No token counts. No brackets after the last sentence. The output ends with the last sentence of the scene. Begin immediately with the video style or shot description."""

    _PREAMBLE_RE = re.compile(
        r"^(Sure!?|Certainly!?|Absolutely!?|Of course!?|Here(?:'s| is).*?:|Great!?|"
        r"LTX-?2(?:\.\d)?(?:\s+\w+)*\s*prompt\s*:|Prompt\s*:|Output\s*:|Scene\s*:)[^\n]*\n?",
        re.IGNORECASE,
    )
    _ROLE_BLEED_RE = re.compile(
        r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$",
        re.IGNORECASE,
    )

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded_model_key = None
        self._last_portrait = False
        self._last_style = ""

    def load_model(self, model_key: str, offline_mode: bool, local_path: str):
        if self.model is not None and self.loaded_model_key != model_key:
            print(f"[LTX2] Model switch detected: {self.loaded_model_key} → {model_key}")
            self.unload_model()

        if self.model is not None:
            return

        if offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            print("[LTX2] Offline mode ON — no network calls will be made.")
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            print("[LTX2] Offline mode OFF — will download if needed.")

        hf_model_id = self.MODELS[model_key]

        if local_path.strip():
            model_source = local_path.strip()
            print(f"[LTX2] Using local path: {model_source}")
        elif offline_mode:
            model_source = hf_model_id
            print(f"[LTX2] Using HF cache for: {hf_model_id}")
        else:
            print(f"[LTX2] Auto-downloading if needed: {hf_model_id}")
            try:
                from huggingface_hub import snapshot_download
                model_source = snapshot_download(hf_model_id)
                print(f"[LTX2] Model ready at: {model_source}")
            except Exception as e:
                print(f"[LTX2] snapshot_download failed, falling back to model ID: {e}")
                model_source = hf_model_id

        print(f"[LTX2] Loading: {model_key}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=offline_mode,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=offline_mode,
        )

        self.model.config.use_cache = True
        self.model.eval()
        self.loaded_model_key = model_key
        print(f"[LTX2] Loaded: {model_key}")

    def unload_model(self):
        """
        Hard VRAM free — equivalent to ComfyUI right-click Free Memory.
        Does NOT just move to CPU (that leaves the reserved block sitting).
        Destroys tensors in place, resets the CUDA allocator, and tells
        ComfyUI's model manager to also drop whatever it is holding.
        """
        if self.model is not None:
            # Destroy every tensor in place so CUDA allocator releases pages
            try:
                for _name, module in list(self.model.named_modules()):
                    for _pname, param in list(module.named_parameters(recurse=False)):
                        try:
                            param.data = torch.empty(0)
                        except Exception:
                            pass
                    for _bname, buf in list(module.named_buffers(recurse=False)):
                        try:
                            module._buffers[_bname] = None
                        except Exception:
                            pass
            except Exception as e:
                print(f"[LTX2] Tensor destroy warning: {e}")

        # Delete Python references
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass

        self.model = None
        self.tokenizer = None
        self.loaded_model_key = None

        # Triple gc — catches circular refs from transformers internals
        gc.collect()
        gc.collect()
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                # Reset the caching allocator entirely — this is what actually
                # releases the "reserved but not allocated" block ComfyUI shows
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[LTX2] CUDA flush warning: {e}")

        # Tell ComfyUI model manager to drop everything it is holding too
        # (same calls ComfyUI makes on Free Memory / Unload Models)
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
            print("[LTX2] ComfyUI mm.unload_all_models + soft_empty_cache done.")
        except Exception as e:
            print(f"[LTX2] ComfyUI mm call skipped: {e}")

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved  = torch.cuda.memory_reserved()  / 1024**3
                print(f"[LTX2] VRAM after free: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")
            except Exception:
                pass
        else:
            print("[LTX2] Model unloaded (no CUDA).")

    @staticmethod
    def _clean_output(text: str) -> str:
        text = text.strip()

        # Strip Qwen3 thinking blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 1. Strip leading preamble
        text = LTX2PromptArchitect._PREAMBLE_RE.sub("", text)

        # 2. Strip trailing role bleed
        text = LTX2PromptArchitect._ROLE_BLEED_RE.sub("", text)

        # 3. Strip inline role injections between sentences
        text = re.sub(
            r"\.(assistant|user|system|<\|[^|>]*\|>)\s*\n",
            ".\n",
            text,
            flags=re.IGNORECASE,
        )

        # 4. Strip trailing Note: blocks
        text = re.sub(r"\s*\n+Note:.*$", "", text, flags=re.DOTALL).strip()

        # Strip AMBIENT tag leftovers
        ambient_match = re.search(r"\[AMBIENT:[^\]]*\]", text, flags=re.IGNORECASE)
        if ambient_match:
            text = text[:ambient_match.end()].strip()

        text = re.sub(r"\s*\(Lora:[^)]*\)\s*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*\(Note:.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"[\s)]{3,}$", "", text).strip()

        text = re.sub(
            r"\s*\n+\d+\s+tokens[\s,].*$", "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
        text = re.sub(
            r"\s*\n+(Please let me know|Let me revise|No further revision|Confirmed\.|"
            r"Written to meet|The scene is now over|The output ends|The task is|The task was|"
            r"The goal was|Nothing more|No continuation|No additional|The response does not|"
            r"It does not continue|It ceases when|Any such statement|"
            r"Output length:|Action count:|Total time:|Last character:|I avoided|I wrote|"
            r"I adhered|I hope this|Thank you for your|Please confirm|I submitted|"
            r"I can revise|feel free to instruct).*$",
            "", text, flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        text = re.sub(
            r"\s*(Ended\.\s*\d+\s*actions|"
            r"\d+\s+actions[\.,]\s*\d+\s+tokens|"
            r"\d+\s+tokens[\.,]\s*Done|"
            r"Done\.\s+\d+\s+seconds|"
            r"Finished\.\s+\d+|"
            r"The end\.\s+\d+\s+seconds|"
            r"Fading to black\.\s+The end|"
            r"The model stops|The output ends here|The scene ends here|"
            r"It\'s complete now|All done\.|Stop now\.|"
            r"End of prompt|End of output|No more to add|Nothing to revise|"
            r"The work is (?:done|finished|complete)|The prompt is (?:done|finished|complete)|"
            r"No further writing|No more writing|Stop\.\s+Finish|Finished\.\s+Complete|"
            r"The scene is complete|The scene is over|Complete\.\s+Finished|"
            r"Hard stop\.\s+End\.|Hard stop\.\s+The end\.|"
            r"Hard stop\.\s+End of scene\.|"
            r"The camera does not move again\.|"
            r"The man is gone\.|The woman is gone\.|"
            r"The market continues without (?:him|her|them)\.|"
            r"No more\.\s+Silence\.|End\.\s+No more\.|"
            r"a (?:stark|final) reminder of the .{5,60} style|"
            r"a testament to the .{5,60} attention to detail|"
            r"Done\.\s+No more|BorderSide:|"
            r"\(End of scene\)|\(End of Scene\)|End of scene\.|End of Scene\.)",
            "", text, flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        text = re.sub(r"(\s*\b(\w)\b\s*){10,}", " ", text).strip()
        text = re.sub(r"\s*\(\d+\s+tokens?[^)]*\)", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*(\([^)]{5,120}\)\s*){2,}$", "", text, flags=re.DOTALL).strip()
        text = re.sub(
            r"\s*\([^)]{0,200}(no setup|no resolution|action count|actions adhered|"
            r"token count|pacing|dialogue integrated|character age|inline prose|"
            r"no padding|no extraneous|exactly \d+ action|hard stop|BorderSide)[^)]{0,200}\)\s*$",
            "", text, flags=re.IGNORECASE | re.DOTALL,
        ).strip()

        text = re.sub(r"\(Exact timing:.*?\)", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"\s*\n*(token|word)\s+count\s*:\s*\d+.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"\[TIME LIMIT[^\]]*\]", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\[PACING[^\]]*\]", "", text, flags=re.IGNORECASE).strip()

        # Catch pacing instruction bleed — model echoing "Hard stop." or token counts mid-prose
        text = re.sub(r'\.\s+Hard stop\..*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\s+Hard stop\..*$', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\.\s+\d+\s+tokens?\b.*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\.\s+\d+\s+words?\b.*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        # Catch "The total duration of the scene is X seconds..." summary bleed
        text = re.sub(r'\.?\s+The total duration of the scene.*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r'\.?\s+The (scene\'?s? )?total (duration|running time).*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r',?\s+with\s+(three|two|four|five|\d+)\s+distinct\s+actions.*$', '.', text, flags=re.DOTALL | re.IGNORECASE).strip()
        text = re.sub(r"\s*\(\d+\s+seconds?\)\s*$", "", text).strip()
        text = re.sub(r"\s*\(\d+:\d+\s*[-–]\s*\d+:\d+\)\s*", " ", text).strip()
        text = re.sub(r"\(The action takes up roughly[^\)]*\)", " ", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\((?:DOWN|UP|PULL|PUSH|ZOOM|HOLD|FADE|PAN|TILT|TRUCK|DOLLY|AMBIENT)[^\)]{0,80}\)", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\[AMBIENT:\s*([^\]]*)\]", r"\1", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Strip inline parenthetical annotation leaks e.g. (camera angle: bird's-eye), (genre: nature, style: drone)
        text = re.sub(r"\([a-z][a-z ,]+:[^)]{3,100}\)", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s{2,}", " ", text).strip()

        # Strip instruction label echoes
        text = re.sub(
            r"^(Action Beat \d+:|Undressing Segment:|Flash/Reveal Segment:|Note:|Scene Instruction:|Pacing:|Dialogue Instruction:).*",
            "", text, flags=re.IGNORECASE | re.MULTILINE,
        ).strip()

        # Strip trailing lone bracket
        text = re.sub(r'\s*[\(\[]\s*$', '', text).strip()

        # Catch "The scene ends there" leaking mid-prose after a sentence
        text = re.sub(r'\.\s+The scene ends there[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+before the scene fades to black[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\.?\s+[Tt]he scene fades to black[^.]*\.', '.', text, flags=re.IGNORECASE).strip()
        text = re.sub(r',?\s+as the scene fades[^.]*\.', '.', text, flags=re.IGNORECASE).strip()

        # ── Repetition loop detection ─────────────────────────────────────────
        # Catches runaway "No more. No more. No more." style loops
        # Safe approach: detect a short phrase repeated 5+ times at end of text
        rep_match = re.search(
            r'((?:\b\w[\w\'\-]*\b[\s\.,!?]*){1,5})\1{4,}$',
            text, flags=re.DOTALL
        )
        if rep_match:
            text = text[:rep_match.start()].strip()

        return text.strip()

    def _build_stop_token_ids(self) -> list:
        delimiter_strings = [
            "assistant", "user", "system",
            "<|eot_id|>", "<|end_of_turn|>", "<|im_end|>",
            "<end_of_turn>", "[/INST]", "### Human", "### Assistant",
        ]
        stop_ids = [self.tokenizer.eos_token_id]
        for s in delimiter_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[0])
        seen = set()
        unique = []
        for tid in stop_ids:
            if tid is not None and tid not in seen:
                seen.add(tid)
                unique.append(tid)
        print(f"[LTX2] Stop token IDs: {unique}")
        return unique

    def generate(
        self,
        bypass, user_input, creativity, seed, invent_dialogue,
        keep_model_loaded, offline_mode, frame_count, model,
        local_path_8b, local_path_3b,
        style_preset="None — let the LLM decide",
        portrait_mode=False,
        scene_context="",
        lora_triggers="",
    ):
        # ── Bypass mode ──────────────────────────────────────────────────────
        if bypass:
            print("[LTX2] Bypass ON — skipping model, passing user_input directly.")
            if self.model is not None and not keep_model_loaded:
                print("[LTX2] Bypass: cleaning up leftover model from cancelled run.")
                self.unload_model()
            neg_prompt = _build_negative_prompt("", user_input, is_portrait=portrait_mode, style_preset=style_preset)
            fps = self.PRESET_FPS.get(style_preset, 24)
            return (user_input.strip(), user_input.strip(), neg_prompt, fps, "")

        # ── Pre-run VRAM clear — always runs before loading anything ────────────
        # Clears whatever the previous cancelled/completed video generation left
        # behind. Runs unconditionally so the LLM never loads on top of stale VRAM.
        if self.model is not None and self.loaded_model_key != model:
            print(f"[LTX2] Model mismatch — unloading stale model before reload.")
            self.unload_model()

        # Tell ComfyUI to release any models IT is holding before we load the LLM
        # This is the key step — clears the LTX video model from VRAM first
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            mm.soft_empty_cache()
            print("[LTX2] Pre-run: ComfyUI models unloaded.")
        except Exception as e:
            print(f"[LTX2] Pre-run mm call skipped: {e}")

        if torch.cuda.is_available():
            try:
                gc.collect()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                reserved_gb  = torch.cuda.memory_reserved()  / 1024**3
                print(f"[LTX2] Pre-run VRAM: {allocated_gb:.2f}GB allocated / {reserved_gb:.2f}GB reserved")
            except Exception as e:
                print(f"[LTX2] Pre-run CUDA flush warning: {e}")

        path_map = {
            "8B - NeuralDaredevil (High Quality)": local_path_8b,
            "3B - Llama-3.2 Abliterated (Low VRAM)": local_path_3b,
        }
        local_path = path_map.get(model, "")
        self.load_model(model_key=model, offline_mode=offline_mode, local_path=local_path)

        # ── Style preset + portrait ───────────────────────────────────────────
        preset_data            = self.STYLE_PRESETS.get(style_preset, ("", False))
        style_instruction_text = preset_data[0]
        is_portrait            = portrait_mode or preset_data[1]

        # Labels that must appear verbatim at the start of the generated prompt
        # so LTX-2's text encoder knows the render style
        PRESET_STYLE_LABEL = {
            # Cinematic
            "Slow-burn thriller":                       "Slow-burn psychological thriller.",
            "Handheld documentary":                     "Handheld documentary footage.",
            "High fashion editorial":                   "High fashion editorial video.",
            "Noir — deep shadows, venetian light":      "Classic noir, black and white, venetian blind shadows.",
            "Golden hour drama":                        "Golden hour cinematic drama.",
            "Horror — desaturated, harsh contrast":     "Horror film, desaturated, harsh contrast.",
            # Adult
            "Erotic cinema — tasteful, cinematic":      "Tasteful erotic cinema, warm intimate lighting.",
            "Explicit — direct, anatomical":            "Explicit adult video, direct lighting.",
            "Voyeur — handheld, observational":         "Voyeuristic handheld footage.",
            "Softcore editorial — lingerie-adjacent":   "Softcore editorial, fashion magazine aesthetic.",
            "Amateur — naturalistic, raw":              "Amateur home video, naturalistic.",
            # Action
            "Action blockbuster":                       "Action blockbuster, teal and orange grade.",
            "Sports documentary":                       "Sports documentary footage.",
            "Music video — stylised":                   "Stylised music video.",
            # Aesthetic
            "Lo-fi home video — VHS":                   "Lo-fi VHS home video footage.",
            "Hyper-real 4K — clinical sharpness":       "Hyper-real 4K, clinical sharpness.",
            "Dreamy — soft focus, slow motion":         "Dreamy soft focus, slow motion.",
            "Gritty realism — flat, natural light":     "Gritty realism, flat natural light.",
            # Speciality
            "POV — first person, immersive":            "First-person POV footage.",
            "Portrait vertical — 9:16 mobile":          "Vertical 9:16 mobile video.",

            # Animation
            "Anime — Japanese animation":               "Japanese anime animation, hand-drawn cel style.",
            "2D cartoon — hand-drawn":                  "2D hand-drawn cartoon animation.",
            "3D CGI — Pixar/DreamWorks":                "3D CGI animation, Pixar style.",
            "Stop motion — claymation":                 "Stop motion claymation animation.",
            "Comic book / graphic novel":               "Comic book graphic novel style.",
            "Cel-shaded — flat colour 3D":              "Cel-shaded 3D animation, flat colour fills.",
            "Rotoscope — animated over live action":    "Rotoscoped animation over live action.",
            "Cyberpunk neon illustrated":               "Cyberpunk neon illustrated, magenta and cyan.",
            "Sci-fi — cinematic, practical":            "Cinematic science fiction, practical sets.",
        }

        style_label = PRESET_STYLE_LABEL.get(style_preset, "")

        if style_instruction_text:
            style_instruction = (
                f"\n[STYLE INSTRUCTION — MANDATORY AESTHETIC ANCHOR: {style_instruction_text} "
                f"Every aspect of the output — lighting, camera, colour, pacing, mood — must reflect this style. "
                f"CRITICAL: You MUST begin your output with exactly these words: \"{style_label}\" — "
                f"then continue with the scene description. This label must be the very first words of your output "
                f"so the video model knows what render style to use. Do not deviate from this style.]"
            )
            print(f"[LTX2] Style preset: {style_preset} → label: {style_label}")
        else:
            style_instruction = ""

        if is_portrait:
            portrait_instruction = (
                "\n[PORTRAIT MODE — MANDATORY: This is a 9:16 vertical video for mobile. "
                "All framing must be vertical — tight head-to-torso shots. "
                "No wide horizontal establishing shots. Action moves vertically in frame. "
                "Camera stays close. Optimised for TikTok, Reels, Shorts.]"
            )
            print("[LTX2] Portrait mode ON")
        else:
            portrait_instruction = ""

        self._last_portrait = is_portrait
        self._last_style    = style_preset

        # ── Timing & pacing ───────────────────────────────────────────────────
        real_seconds = frame_count / 24.0
        action_count = max(1, min(10, round(real_seconds / 4)))

        if action_count == 1:
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY 1 action. One single moment. "
                f"Do not describe anything before or after it. No setup, no resolution. "
                f"HARD STOP after the 1st action. Do not continue."
            )
        else:
            ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY {action_count} distinct actions — NO MORE THAN {action_count}. "
                f"Each action takes roughly {real_seconds / action_count:.0f} seconds of screen time. "
                f"Do not add setup, backstory, or resolution beyond these {action_count} actions. "
                f"Dialogue is woven into action beats — it does not consume a beat and does not replace physical action. "
                f"HARD STOP after the {ordinal} action is complete. The scene ends there. Do not write a {action_count + 1}th action under any circumstances."
            )

        # ── Seed ──────────────────────────────────────────────────────────────
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ── Dynamic token budget ─────────────────────────────────────────────
        # LTX-2.3 text encoder effectively uses ~200 words max.
        # Anything beyond 500 words is wasted — the model ignores the tail.
        # Target scales with clip length but is hard-capped at 500.
        # The LLM generation ceiling is 2x the target so it never clips mid-sentence.
        LTX_WORD_FLOOR   = 150   # minimum — enough detail for any clip
        LTX_WORD_CEILING = 500   # hard cap — LTX-2.3 doesn't use beyond this

        # Scale: ~80 words per action, clamped to floor/ceiling
        token_val        = max(LTX_WORD_FLOOR, min(LTX_WORD_CEILING, action_count * 80 + 100))
        max_tokens_actual = token_val * 2    # LLM hard stop — always 2x target so it finishes
        min_tokens       = int(token_val * 0.5)
        print(f"[LTX2] Token budget: {token_val} words target / {max_tokens_actual} LLM max (actions={action_count}, {real_seconds:.0f}s)")

        # ── Temperature ───────────────────────────────────────────────────────
        temp_map = {
            "0.5 - Strict & Literal":      0.5,
            "0.8 - Balanced Professional": 0.8,
            "1.0 - Artistic Expansion":    1.0,
        }
        temperature = temp_map[creativity]

        stop_token_ids = self._build_stop_token_ids()

        # ── Content tier detection ────────────────────────────────────────────
        _explicit_re = re.compile(
            r"\b(pussy|cock|dick|penis|vagina|clit|clitoris|anus|asshole|"
            r"tits|cum|orgasm|fuck|fucking|blowjob|handjob|penetrat\w*|"
            r"thrust\w*)\b",
            re.IGNORECASE,
        )
        _sensual_re = re.compile(
            r"\b(naked|nude|topless|undress\w*|strip\w*|takes?\s+off|"
            r"removes?\s+(her|his|their|the)?\s*\w*\s*"
            r"(shirt|dress|top|bra|pants|jeans|clothes|clothing|outfit|underwear|skirt|jacket|coat|robe)|"
            r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|"
            r"shed\w*\s+(her|his|their)?\s*(clothes|clothing|shirt|dress)|"
            r"lift\w*\s+(her|his|their|the)?\s*(shirt|top|dress|skirt|crop|tee|t-shirt)|"
            r"(shirt|top|dress|skirt|crop|tee|t-shirt)\s+(up|lifted|raised|hiked)|"
            r"flash\w*\s+(her|his|their)?\s*(breasts?|chest|tits?|boobs?)|"
            r"sensual|erotic|intimate|lingerie|bare\s+skin|bare\s+body|"
            r"babydoll|nighty|nightie|negligee|corset|bodysuit|thong|g-string|"
            r"sheer|see-through|tease|teasing|seductive|seduce|"
            r"flirt\w*|provocative|suggestive|alluring)\b",
            re.IGNORECASE,
        )
        _undress_re = re.compile(
            r"\b(undress\w*|strip\w*|takes?\s+off|"
            r"removes?\s+(her|his|their|the)?\s*\w*\s*"
            r"(shirt|dress|top|bra|pants|jeans|clothes|clothing|outfit|underwear|skirt|jacket|coat|robe)|"
            r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|"
            r"shed\w*\s+(her|his|their)?\s*(clothes|clothing|shirt|dress)|"
            r"lift\w*\s+(her|his|their|the)?\s*(shirt|top|dress|skirt|crop|tee|t-shirt)|"
            r"(shirt|top|dress|skirt|crop|tee|t-shirt)\s+(up|lifted|raised|hiked)|"
            r"flash\w*\s+(her|his|their)?\s*(breasts?|chest|tits?|boobs?))\b",
            re.IGNORECASE,
        )
        # Detect lift/flash specifically — partial reveal, not full removal
        _lift_re = re.compile(
            r"\b(lift\w*\s+(her|his|their|the)?\s*(shirt|top|dress|skirt|crop|tee|t-shirt)|"
            r"(shirt|top|dress|skirt|crop|tee|t-shirt)\s+(up|lifted|raised|hiked)|"
            r"flash\w*\s+(her|his|their)?\s*(breasts?|chest|tits?|boobs?)|"
            r"hik\w*\s+(her|his|their|the)?\s*(shirt|top|skirt|dress))\b",
            re.IGNORECASE,
        )

        is_explicit    = bool(_explicit_re.search(user_input))
        is_sensual     = bool(_sensual_re.search(user_input)) and not is_explicit
        has_undressing = bool(_undress_re.search(user_input))
        has_lift       = bool(_lift_re.search(user_input))

        # ── Detect exactly which garments the user named ──────────────────────
        _garment_re = re.compile(
            r"\b(shirt|top|blouse|dress|bra|pants|jeans|underwear|skirt|jacket|"
            r"coat|robe|lingerie|clothes|clothing|outfit|thong|g-string|bodysuit|"
            r"corset|nighty|nightie|negligee|babydoll)\b",
            re.IGNORECASE,
        )
        named_garments = list(dict.fromkeys(
            m.group(0).lower() for m in _garment_re.finditer(user_input)
        ))
        garment_list = ", ".join(named_garments) if named_garments else "whatever the user described"

        if is_explicit:
            explicit_instruction = (
                "\n[INSTRUCTION: This is an explicit adult scene. "
                "You MUST use direct anatomical terms — pussy, cock, anus, clit, ass — exactly as written. "
                "Do not replace them with euphemisms like 'womanhood', 'manhood', 'core', or 'sex'. "
                "If the user said 'pussy', write 'pussy' every time. "
                "\n\nSCOPE HARD LIMIT — THIS IS ABSOLUTE: "
                "You may ONLY describe what the user explicitly wrote. "
                "Do NOT add any sexual acts, nudity, or body part exposure the user did not state. "
                "The user's words are the ceiling — you cannot go above them. "
                "If they asked for undressing only, describe only the undressing. "
                "If they asked for one garment removed, remove only that garment. "
                "Do NOT continue to the next logical step. Do NOT improvise what comes next. "
                "The scene ends exactly where the user's request ends. Hard stop. "
                "\n\nUNDRESSING — if the subject starts clothed, use the correct physical sequence for each garment: "
                "Shirt/t-shirt/crop top: grip the hem → lift past stomach → past ribs → over chest → over head → off arms. "
                "SHIRT LIFT (partial — not full removal): fingers find the hem at the waist → grip the fabric → slowly gather and lift → fabric rises past the stomach → past the navel → past the ribs → chest comes into view → breasts fully exposed → held there. Each of these is its own sentence. Do NOT compress into one line. "
                "Dress (zip): find the zip → pull it down slowly → fabric parts → slipped off shoulders → slides down → falls. "
                "Dress (pullover): grip hem at thighs → lift past hips → past waist → over chest → over head. "
                "Blouse/button-down: work each button one at a time → fabric parts → shrug off shoulders → slides down arms. "
                "Bra: reach behind to clasp → unhook → straps off each shoulder → cups fall away. "
                "Jeans/trousers: button popped → zip down → pushed over hips → down the thighs → stepped out of. "
                "Underwear: thumbs into waistband → pushed down → stepped out of. "
                "Each step is its own sentence. Camera lingers on each reveal. Do not compress or skip any step. "
                "NO INVENTED RESOLUTION: Do NOT have the subject lower, cover, or reverse any action unless the user asked for it. If she lifts her shirt, it stays lifted. Do not write her pulling it back down. "
                "Always state character age as a specific number.]"
            )
        elif is_sensual:
            if has_undressing:
                undress_clause = (
                    f"\n\nUNDRESSING SCOPE — ABSOLUTE HARD LIMIT: "
                    f"The user named ONLY these garments: {garment_list}. "
                    f"You may ONLY describe the removal of THOSE specific items — nothing else. "
                    f"Removing ANY other garment — even if it feels like the logical next step — is a scope violation. "
                    f"Do NOT go from a shirt to a bra unless the user said bra. "
                    f"Do NOT go from a bra to topless nudity unless the user said nude or naked or topless. "
                    f"Do NOT go from clothing to underwear unless the user said underwear. "
                    f"Do NOT go from underwear to nudity unless the user said nude or naked. "
                    f"The named garments are the ceiling — you stop there, no matter what the style preset is. "
                    f"\n\nFor each named garment use the correct physical sequence — every step its own sentence: "
                    f"Shirt/t-shirt/crop top (full removal): fingers grip the hem at the waist → fabric lifted past the stomach → past the ribs → over the chest → pulled over the head → off the arms. "
                    f"SHIRT LIFT (partial — not removed): fingers find the hem → grip the fabric → slowly gather and lift → fabric rises past the stomach → past the navel → past the ribs → chest and breasts come into view → held there. Every step is its own sentence. Do NOT compress into one line. "
                    f"Dress (zip): hand finds the zip → pulled slowly down → fabric loosens and parts → slipped off shoulders → slides down the body → falls. "
                    f"Dress (pullover): hands grip hem at thighs → lifted past hips → past waist → gathered over chest → pulled over head. "
                    f"Blouse/button-down: each button worked one at a time → fabric parts → shrugged off shoulders → slides down arms. "
                    f"Bra: hand reaches behind to clasp → unhooked → straps off each shoulder in turn → cups fall away. "
                    f"Jeans/trousers: button popped → zip drawn down → pushed over hips → down the thighs → stepped out of. "
                    f"Underwear: thumbs hooked into waistband → pushed down → stepped out of. "
                    f"Camera lingers on each reveal. Then STOP after the last named garment. "
                    f"NO INVENTED RESOLUTION: Do NOT have the subject lower, cover, or reverse any action unless the user explicitly asked for it. "
                    f"Bare skin and curves may be described naturally — but genitals are never described or zoomed in on."
                )
            else:
                undress_clause = (
                    "\n\nNO UNDRESSING: The user has not asked for any clothing removal. "
                    "Do NOT have the subject remove, loosen, or adjust any clothing in a sexualised way. "
                    "Do NOT describe underwear, bare skin below the neck, or any implied nudity. "
                    "Keep the scene exactly at the level of sensuality the user described — no further."
                )
            explicit_instruction = (
                "\n[INSTRUCTION: This scene involves sensual or intimate content. "
                "Tone: warm, cinematic, tasteful. Bare skin above the waist may be described naturally where the user asked for it. "
                "SCOPE HARD LIMIT — THIS IS ABSOLUTE: "
                "Describe ONLY what the user asked for. Do NOT self-escalate under any circumstances. "
                "Do NOT invent undressing, nudity, or intimate acts the user did not write. "
                "The style preset sets the aesthetic — it does NOT give permission to add content beyond the user's request. "
                "Always state character age as a specific number. "
                + undress_clause + "]"
            )
        else:
            explicit_instruction = (
                "\n[INSTRUCTION: Write a full cinematic LTX-2.3 video prompt. "
                "LTX-2.3 rewards specificity and complexity — do not simplify. "
                "Cover in order: "
                "(1) video style and genre, "
                "(2) camera orientation if subject faces away — state it first, "
                "(3) shot type and camera angle with exact lens specs e.g. '85mm f/1.4', "
                "(4) character — age as a specific number always, hair texture, skin tone, body type, "
                "clothing described with fabric and material e.g. 'a loose cotton shirt' not just 'a shirt', "
                "(5) spatial blocking — where subjects are in frame relative to each other and camera, left/right/fore/background, "
                "(6) scene — location, lighting quality and direction, surface textures and material detail, "
                "(7) action — use VERBS. State who moves, what moves, how they move, what the camera does. "
                "If the scene is static, add environmental motion: wind in hair, background figures, fabric moving. Static prompts freeze. "
                "(8) texture in motion — describe how materials behave: fabric pulling, hair lifting, leather creasing, "
                "(9) camera movement as prose verbs only — no bracketed directions, "
                "(10) sound — physical, concrete, max 2 per beat, tone and intensity described.]"
            )

        # ── Camera orientation detection ──────────────────────────────────────
        # LTX has a strong bias toward front-facing subjects. When the input
        # implies the subject should NOT be facing the camera, we detect it and
        # inject an explicit orientation instruction that fires early in the prompt.
        _facing_away_re = re.compile(
            r"\b(from behind|from the back|rear view|back view|"
            r"watches? her from behind|follows? her|following her|"
            r"walks? away|walking away|moving away|"
            r"back of her|back of his|back of their|"
            r"over her shoulder|over his shoulder|"
            r"she walks|he walks|they walk).{0,40}"
            r"(away|off|past|through|down|out|forward|ahead)\b|"
            r"\b(from behind|rear.?view|back.?view|over.{0,10}shoulder|"
            r"follows? (her|him|them)|watches? (her|him|them) (walk|move|go|leave|pass))\b",
            re.IGNORECASE,
        )
        _facing_camera_re = re.compile(
            r"\b(faces? (the )?camera|looks? (at|into) (the )?camera|"
            r"faces? forward|faces? front|toward (the )?camera|"
            r"selfie|mirror selfie|talking to camera|front.?facing)\b",
            re.IGNORECASE,
        )
        is_facing_away  = bool(_facing_away_re.search(user_input))
        is_facing_camera = bool(_facing_camera_re.search(user_input))

        # Also force facing-away for voyeur preset unless user explicitly said facing camera
        if style_preset == "Voyeur — handheld, observational" and not is_facing_camera:
            is_facing_away = True

        if is_facing_away and not is_facing_camera:
            voyeur_height = (
                " The camera is held at hip or chest height — low and discreet, not raised for a clean shot."
                if style_preset == "Voyeur — handheld, observational" else ""
            )
            orientation_instruction = (
                "\n\n[CAMERA ORIENTATION — CRITICAL: "
                "The subject MUST NOT face the camera at any point in this scene. "
                "She faces AWAY from the camera for the entire duration. "
                "The camera sees her back, the back of her head, and the rear of her body."
                + voyeur_height +
                " BEGIN your output with the camera orientation — e.g. 'Rear view.' or 'The camera follows her from behind.' — "
                "this must be the very first thing stated so the model anchors on it. "
                "No front-facing shots. No over-the-shoulder shots that show her face. "
                "The subject is NEVER seen from the front.]"
            )
        else:
            orientation_instruction = ""

        # ── Sequence detection ────────────────────────────────────────────────
        _sequence_re = re.compile(r"^\s*(\d+[\.\):])\s+.+", re.MULTILINE)
        sequence_steps = _sequence_re.findall(user_input)
        if len(sequence_steps) >= 2:
            step_count = len(sequence_steps)
            sequence_instruction = (
                f"\n[SEQUENCE INSTRUCTION: The user has provided {step_count} numbered steps. "
                f"You MUST follow them in exact order — step 1 first, then step 2, and so on. "
                f"Do not reorder, skip, or merge steps. Each step is one distinct beat in the scene. "
                f"Do not add actions before step 1 or after step {step_count}.]"
            )
        else:
            sequence_instruction = ""

        # ── Anti-static detection ─────────────────────────────────────────────
        # LTX-2.3 produces freeze frames from static prompts. Detect inputs that
        # describe a pose/state with no motion and inject a motion reminder.
        _motion_re = re.compile(
            r"\b(walk\w*|run\w*|mov\w*|turn\w*|lift\w*|bend\w*|reach\w*|pull\w*|push\w*|"
            r"danc\w*|jump\w*|climb\w*|fall\w*|drop\w*|sit\w*|stand\w*|rise\w*|lean\w*|"
            r"nod\w*|shak\w*|wave\w*|stir\w*|pour\w*|open\w*|clos\w*|look\w*|glanc\w*|"
            r"strip\w*|undress\w*|remov\w*|lift\w*|hike\w*|unzip\w*|unbutton\w*|"
            r"crawl\w*|kneel\w*|stretch\w*|sway\w*|bounce\w*|grind\w*|thrust\w*|"
            r"follows?|tracking|panning|dolly|zoom\w*|tilt\w*|orbit\w*|drift\w*)\b",
            re.IGNORECASE,
        )
        has_motion = bool(_motion_re.search(user_input))
        if not has_motion:
            static_instruction = (
                "\n\n[ANTI-STATIC INSTRUCTION: The user\'s input describes a static state with no explicit motion. "
                "LTX-2.3 will freeze on static prompts. You MUST add natural environmental or physical motion to prevent this. "
                "Choose motion that fits the scene without contradicting the user\'s input: "
                "wind moving hair or fabric, the subject\'s breathing visible in their chest, "
                "a subtle weight shift or micro-movement, background figures passing, "
                "leaves or curtains stirring, a light source flickering, the camera drifting slightly. "
                "Keep it subtle — do not invent actions the user explicitly excluded. "
                "The scene must have something moving at all times.]"
            )
        else:
            static_instruction = ""

        # ── Person detection ──────────────────────────────────────────────────
        _person_re = re.compile(
            r"\b(he|she|his|her|him|they|them|their|man|men|woman|women|girl|girls|boy|boys|guy|guys|"
            r"person|people|couple|figure|character|model|actress|actor|"
            r"someone|anybody|nobody|stranger|friend|lover|wife|husband|"
            r"boyfriend|girlfriend|teenager|teenagers|adult|adults|female|male|blonde|brunette|"
            r"redhead|nude|naked|singer|dancer|performer|athlete|soldier|worker|"
            r"player|nurse|doctor|student|teacher|child|children|kid|kids|crowd|audience)\b",
            re.IGNORECASE,
        )
        has_person = bool(_person_re.search(user_input + " " + scene_context))
        if not has_person:
            no_person_instruction = (
                "\n[SCENE INSTRUCTION: The user has not described any person or character. "
                "Do NOT invent or introduce any human figures, silhouettes, voices, or implied presence. "
                "This is a pure environment or object scene. Write only what the user described — "
                "the setting, objects, light, atmosphere, and motion of non-human elements. "
                "No characters. No 'someone', no 'a figure', no implied human presence of any kind. "
                "No dialogue, no whispers, no voices. Sound is limited to the environment only.]"
            )
        else:
            no_person_instruction = ""

        # ── Multi-subject detection ───────────────────────────────────────────
        _multi_re = re.compile(
            r"\b(two\s+(women|men|people|girls|guys|characters|figures|friends|strangers|colleagues|lovers|siblings|brothers|sisters)|"
            r"both\s+(of\s+them|women|men|girls|guys)|"
            r"(she|he)\s+and\s+(she|he|her|him)|"
            r"(a\s+man\s+and\s+a\s+woman|a\s+woman\s+and\s+a\s+man)|"
            r"(a\s+man\s+and\s+a\s+man|a\s+woman\s+and\s+a\s+woman)|"
            r"couple|trio|they\s+(kiss|touch|embrace|undress|fuck|have))\b",
            re.IGNORECASE,
        )
        has_multi_subject = bool(_multi_re.search(user_input + " " + scene_context))
        if has_multi_subject:
            multi_instruction = (
                "\n[MULTI-SUBJECT INSTRUCTION: This scene has two or more people. "
                "For EACH person establish: their position in the frame (left/right/foreground/background), "
                "their spatial relationship to the other person (facing, beside, behind, above, etc.), "
                "and keep track of who is doing what throughout — never let actions become ambiguous. "
                "When referring back to them use consistent descriptors (e.g. 'the dark-haired woman', "
                "'the taller man') — not just 'she' or 'he' which causes confusion with two subjects.]"
            )
        else:
            multi_instruction = ""

        # ── Music / dance detection ───────────────────────────────────────────
        # When user mentions music, dancing, or a beat-driven scene, the "no musical audio"
        # rule must not suppress music description. Detect and override.
        _music_re = re.compile(
            r"\b(music|song|track|beat|bass|rhythm|danc\w*|club|rave|party|dj|"
            r"playlist|bpm|groove|vibe|concert|gig|perform\w*|sing\w*|singer|"
            r"strip\w*club|pole danc\w*|lap danc\w*)\b",
            re.IGNORECASE,
        )
        has_music = bool(_music_re.search(user_input))

        if has_music:
            music_sound_rule = (
                "SOUND RULE FOR THIS SCENE — MUSIC IS PRESENT: "
                "There is music in this scene — describe it as physical sound with energy, tempo, and texture. "
                "Examples: 'a driving kick drum', 'deep bass pulses through the floor', 'sharp hi-hats tick over a slow groove', "
                "'a warm synth pad swells beneath the mix', 'the track drops into a heavy bass line'. "
                "Describe what a body in the room would physically feel and hear. "
                "Maximum 2 additional ambient sounds alongside the music (crowd, breathing, heels on floor). "
                "Do NOT silence the music. Do NOT describe it as abstract emotion — describe it as physical sound."
            )
        else:
            music_sound_rule = (
                "SOUND RULE: Maximum 2 ambient sounds active at any one time. "
                "Only concrete physical sounds — footsteps, a door, rain, an engine, crowd noise. "
                "No abstract emotional audio. No musical metaphors. No 'tension hums' or 'heartbeat of the city'."
            )

        # ── Dialogue instruction ──────────────────────────────────────────────
        if not has_person:
            dialogue_instruction = ""
        elif invent_dialogue:
            dialogue_instruction = (
                "\n\n[DIALOGUE INSTRUCTION — MANDATORY, CANNOT BE SKIPPED: "
                "You MUST include at least one line of spoken dialogue in this scene. "
                "An output with zero spoken words has failed this instruction. "
                "Invent dialogue that sounds like something a real person would actually say in this exact situation — not a cliché. "
                "Write it as inline prose woven into the action, with attribution and physical delivery, like a novel. "
                "The spoken words sit inside the sentence — never as a floating quote, never as a [DIALOGUE: ...] tag. "
                "Examples: "
                "'\"Don\\'t stop,\" she breathes, gripping the sheets, her voice barely above a whisper.' "
                "'She glances back, \"Are you watching me?\" her tone half-amused, half-serious.' "
                "'\"Come here,\" he says quietly, his hand extended.' "
                "If the scene is sexual or explicit, dialogue must reflect that — breathless, reactive, direct. "
                "Weave it into a physical beat — the character speaks while doing something, not in a static pause. "
                + music_sound_rule + "]"
            )
        else:
            has_user_dialogue = bool(re.search(r'["\u201c\u201d]', user_input))
            if has_user_dialogue:
                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION: Use ONLY the dialogue the user provided — do not invent or add any additional spoken words. "
                    "Place their exact words naturally in the scene as inline prose with attribution and delivery. "
                    "Examples: 'She smiles, \"I\\'m so happy,\" her voice bright, eyes wide.' "
                    "Never use [DIALOGUE: ...] tags. Weave the words into the action as part of the prose.]"
                )
            else:
                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION: No dialogue in this scene. No spoken words. "
                    "Weave sound naturally into the prose instead — described as prose, not tags. "
                    + music_sound_rule + "]"
                )

        # ── Length instruction ────────────────────────────────────────────────
        length_instruction = (
            f"\n[PACING: {pacing_hint} "
            f"Aim for approximately {token_val} words of prose. "
            f"Do not exceed the action count above. "
            f"Output ends with the final sentence of the scene — no summaries, no counts, no notes, no brackets after the last word.]"
        )

        # ── Lift / flash instruction — fires when a shirt lift or flash is detected ──
        if has_lift:
            lift_instruction = (
                "\n\n[SHIRT LIFT INSTRUCTION — THIS OVERRIDES ALL OTHER UNDRESSING GUIDANCE FOR THIS ACTION: "
                "The user has described a shirt, top, or crop top being lifted. "
                "You MUST write this as a sequence of separate sentences — one sentence per step. "
                "DO NOT compress the lift into a single sentence. DO NOT write 'she lifts her shirt, revealing her breasts' as one line. "
                "MANDATORY SEQUENCE — write each of these as its own sentence in the output:\n"
                "1. Her fingers find the hem of her shirt at the waist.\n"
                "2. She grips the fabric and begins to gather it upward.\n"
                "3. The shirt rises slowly past her stomach.\n"
                "4. The fabric passes her navel, exposing her bare midriff.\n"
                "5. The shirt climbs past her ribs.\n"
                "6. Her chest comes into view as the fabric rises higher.\n"
                "7. Her breasts are fully exposed, the shirt held up.\n"
                "The shirt STAYS LIFTED for the remainder of the scene. "
                "Do NOT write her lowering it, covering herself, or adjusting the shirt unless the user explicitly asked for that.]"
            )
        else:
            lift_instruction = ""

        # ── Vision context ────────────────────────────────────────────────────
        if scene_context and scene_context.strip():
            effective_input = (
                f"[SCENE CONTEXT FROM IMAGE — use this as the authoritative description "
                f"of the subject and setting; do not invent or contradict it]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION — apply this as action, style, and mood over the above scene]\n"
                f"{user_input.strip()}"
            )
        else:
            effective_input = user_input.strip()

        # ── LoRA triggers ─────────────────────────────────────────────────────
        if lora_triggers and lora_triggers.strip():
            if style_label:
                lora_instruction = (
                    f"\n[LORA INSTRUCTION: You MUST begin the prompt output with these exact trigger words "
                    f"before anything else: {lora_triggers.strip()} — then immediately follow with \"{style_label}\" "
                    f"then continue with the scene description.]"
                )
            else:
                lora_instruction = (
                    f"\n[LORA INSTRUCTION: You MUST begin the prompt output with these exact trigger words "
                    f"before anything else: {lora_triggers.strip()} — place them as the very first words of your output, "
                    f"then continue with the scene description immediately after.]"
                )
        else:
            lora_instruction = ""

        # ── Build messages ────────────────────────────────────────────────────
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": (
                effective_input
                + orientation_instruction
                + style_instruction
                + portrait_instruction
                + sequence_instruction
                + static_instruction
                + no_person_instruction
                + multi_instruction
                + dialogue_instruction
                + explicit_instruction
                + lift_instruction
                + lora_instruction
                + length_instruction
            )},
        ]

        raw = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self.model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self.model.device)
        elif isinstance(raw, list):
            input_ids = torch.tensor([raw], dtype=torch.long).to(self.model.device)
        else:
            input_ids = raw.to(self.model.device)

        input_length = input_ids.shape[1]

        # ── Generation — wrapped so cancelled runs always unload ──────────────
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    min_new_tokens=min_tokens,
                    max_new_tokens=max_tokens_actual,
                    temperature=temperature,
                    do_sample=True,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.07,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=stop_token_ids,
                )
        except Exception as _gen_exc:
            # Generation cancelled or errored — unload immediately so next run
            # doesn't find a half-dead model occupying VRAM
            print(f"[LTX2] Generation interrupted: {_gen_exc}")
            print("[LTX2] Forcing unload due to interrupted generation.")
            self.unload_model()
            raise

        generated_tokens = output_ids[0][input_length:]
        result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        del output_ids
        del input_ids

        result = self._clean_output(result)

        # ── Style label safety net ────────────────────────────────────────────
        # If the LLM forgot to open with the style label, prepend it now
        # so LTX-2's text encoder always sees the render style
        if style_label and not result.lower().startswith(style_label.split()[0].lower()):
            result = style_label + " " + result
            print(f"[LTX2] Style label prepended (LLM omitted it): {style_label}")

        neg_prompt = _build_negative_prompt(
            result, user_input,
            is_portrait=self._last_portrait,
            style_preset=self._last_style,
        )

        # ── Prompt history ────────────────────────────────────────────────────
        # Saves last 20 runs to prompt_history.json next to this script
        # Also builds a HISTORY string for the output pin (last 5 runs)
        history_string = ""
        try:
            _hist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_history.json")
            _history = []
            _first_save = not os.path.exists(_hist_path)
            if not _first_save:
                with open(_hist_path, "r", encoding="utf-8") as _f:
                    _history = json.load(_f)
            _history.insert(0, {
                "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
                "style":     style_preset,
                "portrait":  is_portrait,
                "input":     user_input[:300],
                "output":    result[:800],
            })
            _history = _history[:20]
            with open(_hist_path, "w", encoding="utf-8") as _f:
                json.dump(_history, _f, indent=2, ensure_ascii=False)

            if _first_save:
                print(f"[LTX2] Prompt history file created at: {_hist_path}")
            else:
                print(f"[LTX2] Prompt history saved ({len(_history)} runs) → {_hist_path}")

            # Build the HISTORY output pin — last 5 runs as readable text
            lines = []
            for i, entry in enumerate(_history[:5], 1):
                lines.append(
                    f"── Run {i}  [{entry['timestamp']}]  {entry['style']}\n"
                    f"IN:  {entry['input'][:120]}\n"
                    f"OUT: {entry['output'][:300]}"
                )
            history_string = "\n\n".join(lines)

        except Exception as _he:
            print(f"[LTX2] Prompt history save failed (non-fatal): {_he}")
            history_string = f"[History unavailable: {_he}]"

        if not keep_model_loaded:
            self.unload_model()

        fps = self.PRESET_FPS.get(style_preset, 24)
        print(f"[LTX2] FPS output: {fps}  (preset: {style_preset})")

        return (result, result, neg_prompt, fps, history_string)


# ── ComfyUI boilerplate ──────────────────────────────────────────────────────

class LTX2UnloadModel:
    """Utility node to manually free VRAM when keep_model_loaded is True."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "LTX2"
    OUTPUT_NODE = True

    def unload(self):
        import gc
        unloaded = 0
        for obj in gc.get_objects():
            if isinstance(obj, LTX2PromptArchitect) and obj.model is not None:
                obj.unload_model()
                unloaded += 1
        print(f"[LTX2] Unload node: freed {unloaded} model instance(s).")
        return {}


NODE_CLASS_MAPPINGS = {
    "LTX2PromptArchitect": LTX2PromptArchitect,
    "LTX2UnloadModel":     LTX2UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2PromptArchitect": "LTX-2.3 Easy Prompt By LoRa-Daddy",
    "LTX2UnloadModel":     "LTX2 Unload Model",
}
