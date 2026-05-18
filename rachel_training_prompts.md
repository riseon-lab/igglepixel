# Rachel LoRA Training Prompt Pack

These prompts are for generating a curated character dataset from a separate identity/reference system. They intentionally avoid describing the character's fixed identity traits such as hair colour, eye colour, face shape, or name. Use the same identity reference for every image, then convert each final image into a matching training caption after generation.

This set is likeness-first: keep the face prominent. The framing avoids distant full-body and wide shots because character likeness degrades as the subject gets smaller in frame.

Caption conversion rule:

```text
Generation prompt: the same adult woman, [scene/outfit/pose/camera/lighting details]
Training caption: A woman named Rachel, [visible scene/outfit/pose/camera/lighting details]
```

Keep only images where the face and body feel like the same person. Reject images where the subject is too far from camera, the face is soft, or the likeness is weaker than the close portraits.

## Portraits

1. `A close-up studio portrait of the same adult woman, neutral expression, shoulders relaxed, clean moonstone grey backdrop, soft key light, subtle fill light, realistic skin texture, 85mm portrait lens, shallow depth of field.`

2. `A close-up portrait of the same adult woman, soft smile, head slightly tilted, simple black top, warm window light from the left, softly blurred apartment background, 85mm lens, face sharp and prominent.`

3. `A tight beauty portrait of the same adult woman, calm direct gaze, minimal natural make-up, white studio backdrop, high-key lighting, crisp editorial finish, face filling most of the frame.`

4. `A close 3/4 view portrait of the same adult woman, serious expression, chin slightly lifted, dark studio background, cinematic rim light, low-key exposure, shallow depth of field.`

5. `A side-profile close-up portrait of the same adult woman, calm expression, elegant posture, black background, narrow strip of side light, fine realistic detail, 100mm portrait lens.`

6. `An over-the-shoulder close portrait of the same adult woman, confident expression, body turned away but face looking back to camera, soft studio light, neutral backdrop, fashion editorial framing.`

7. `A close-up portrait of the same adult woman, eyes looking slightly off-frame, thoughtful expression, soft shadows, muted colour grade, realistic documentary portrait style, face in sharp focus.`

8. `A close candid portrait of the same adult woman, relaxed expression, face framed by soft foreground blur, warm interior lighting, natural photography, 50mm lens.`

9. `A head-and-shoulders portrait of the same adult woman, subtle smirk, clean black turtleneck, simple grey background, balanced studio exposure, sharp realistic detail.`

10. `A dramatic close-up portrait of the same adult woman, intense gaze, face partly in shadow, hard side light, dark background, cinematic contrast, editorial magazine style.`


## Expressions

11. `A tight portrait of the same adult woman laughing naturally, casual white t-shirt, bright daylight, candid street background blurred behind her, natural motion in the shoulders.`

12. `A close portrait of the same adult woman with a pensive expression, looking down, hands lightly clasped near her chin, soft indoor window light, quiet apartment background, muted colours.`

13. `A waist-up portrait of the same adult woman with a confident expression, arms crossed, simple black blazer, clean studio background, strong soft key light, commercial portrait style.`

14. `A close portrait of the same adult woman with a surprised expression, mouth slightly open, one hand near the collar, urban street background softly blurred, natural daylight.`

15. `A close portrait of the same adult woman with a calm expression and closed eyes, face angled toward sunlight, minimal outfit, warm golden hour lighting, peaceful atmosphere.`

16. `A close cafe portrait of the same adult woman with a playful subtle smile, seated at a small table, hands around a coffee cup, warm practical lighting, shallow depth of field.`

17. `A tight editorial portrait of the same adult woman with a serious expression, glossy make-up, black background, flash photography lighting, high contrast fashion campaign style.`

18. `A close portrait of the same adult woman with a focused expression, looking through a window, reflections across the glass, cool blue hour lighting, cinematic mood.`

19. `A head-and-shoulders portrait of the same adult woman with a relaxed smile, outdoors in overcast light, soft natural background blur, documentary portrait framing, realistic detail.`

20. `An intimate close portrait of the same adult woman with a subtle worried expression, seated near a bed, soft morning light, neutral bedroom setting, candid style.`


## Outfits

21. `A waist-up portrait of the same adult woman wearing a black leather jacket, white t-shirt, dark jeans partly visible, city street background blurred, overcast daylight, relaxed posture.`

22. `A close indoor portrait of the same adult woman wearing an oversized grey knit sweater, minimal jewellery, seated near a neutral wall, soft window light, relaxed expression.`

23. `A waist-up studio portrait of the same adult woman wearing a tailored black suit and fitted blazer, confident stance, clean white backdrop, commercial fashion lighting.`

24. `A close lifestyle portrait of the same adult woman wearing a cream cardigan over a white top, warm home interior, soft practical lamp light, gentle expression.`

25. `A medium-close cinematic portrait of the same adult woman wearing a long dark trench coat, collar visible, rainy pavement and neon reflections behind her, night city background.`

26. `A close street portrait of the same adult woman wearing a denim jacket, black trousers partly visible, crossbody bag strap, quiet sidewalk background, natural overcast light.`

27. `A polished studio portrait of the same adult woman wearing a satin blouse and delicate earrings, soft beauty lighting, clean warm grey backdrop, bust framing.`

28. `A close activewear portrait of the same adult woman wearing a fitted athletic jacket, leggings partly visible, morning light, casual confident pose, soft background blur.`

29. `A close fashion portrait of the same adult woman wearing a red satin dress, minimal accessories, luxury hotel lobby softly blurred, warm ambient lighting, 85mm lens.`

30. `A waist-up professional portrait of the same adult woman wearing a white silk shirt and high-waisted black trousers partly visible, minimal office space, soft daylight.`

31. `A close commercial portrait of the same adult woman wearing a metallic silver jacket over a black fitted top, glossy lighting, confident direct gaze, fashion campaign composition.`

32. `A head-and-shoulders portrait of the same adult woman wearing a black turtleneck and wool coat, standing near a stone wall, cool winter daylight, hands near coat lapels.`

33. `A close outdoor portrait of the same adult woman wearing a casual summer dress under a light jacket, quiet garden path behind her, soft afternoon daylight.`

34. `A waist-up studio image of the same adult woman wearing a utility vest over a plain shirt, cargo styling partly visible, relaxed practical styling, neutral expression.`

35. `A close editorial portrait of the same adult woman wearing a velvet blazer, satin camisole, subtle jewellery, dark moody interior, low-key warm lighting.`

36. `A medium-close street portrait of the same adult woman wearing a long wool coat and scarf, windy street corner softly blurred, late afternoon light, face prominent.`


## Pose and Framing

37. `A waist-up studio portrait of the same adult woman standing upright, hands in pockets just visible, relaxed shoulders, simple outfit, clean grey backdrop, even soft lighting.`

38. `A seated close portrait of the same adult woman on a simple wooden chair, one elbow resting on the chair back, calm expression, soft studio light, bust framing.`

39. `A medium-close portrait of the same adult woman walking toward the camera, natural mid-stride shoulder motion, coat moving slightly, city background blurred, 50mm lens.`

40. `A waist-up portrait of the same adult woman leaning against a textured wall, arms crossed, confident expression, side light, shallow depth of field.`

41. `A close three-quarter body crop of the same adult woman turning toward camera mid-step, one hand adjusting her jacket, cinematic street light, face sharp.`

42. `A close over-the-shoulder portrait of the same adult woman, one hand resting near her collar, softly blurred background, editorial fashion framing.`

43. `A close seated portrait of the same adult woman sitting on concrete steps, elbows on knees, relaxed casual pose, urban background blurred, overcast daylight.`

44. `A waist-up studio portrait of the same adult woman standing with one hand on hip, the other hand holding sunglasses near her chest, crisp commercial lighting.`

45. `A dynamic medium-close portrait of the same adult woman crossing a street, long coat moving with the step, urban background blurred, natural motion, documentary style.`

46. `A seated editorial close portrait of the same adult woman on a modern sofa, elegant posture, warm interior lighting, upper-body framing, face prominent.`

47. `A close fashion image of the same adult woman in a contrapposto pose, chin slightly lifted, hands relaxed at frame edge, neutral studio backdrop, polished styling.`

48. `A candid close portrait of the same adult woman crouching slightly to tie a boot, face still visible and sharp, street setting, natural daylight, medium framing.`


## Lighting and Camera

49. `A close-up portrait of the same adult woman in golden hour light, soft bokeh background, warm colour grade, 85mm lens, shallow depth of field.`

50. `A close portrait of the same adult woman with underexposed shadows, dramatic side lighting, dark cinematic background, low-key exposure, film still mood.`

51. `A tight portrait of the same adult woman in bright high-key studio lighting, white background, clean beauty campaign finish, even skin detail, sharp focus.`

52. `A close portrait of the same adult woman in blue hour city light, cool colour grade, blurred street lights behind her, quiet expression, cinematic realism.`

53. `A close portrait of the same adult woman with Aerochrome false-colour infrared look, surreal pink foliage blurred behind her, calm expression, soft outdoor light.`

54. `A close portrait of the same adult woman from a low camera angle, architectural columns softly blurred behind her, confident expression, dramatic perspective, 35mm lens.`

55. `A waist-up portrait of the same adult woman photographed from a high angle, seated at a desk, soft window light, quiet thoughtful mood, 50mm lens.`

56. `A close portrait of the same adult woman lit by neon signs, magenta and cyan highlights, rainy night background blurred, shallow depth of field, cinematic street style.`

57. `A close studio portrait of the same adult woman with hard flash lighting, crisp shadow behind her, simple outfit, editorial magazine look, upper-body framing.`

58. `A close portrait of the same adult woman with soft volumetric light beams through a window, dust in the air, warm interior scene, quiet expression.`

59. `A close candlelight portrait of the same adult woman, warm low-light interior, subtle shadows, intimate cinematic mood, face clear and softly detailed.`

60. `A medium-close portrait of the same adult woman in overcast natural light, soft shadows, muted colour grade, empty street background blurred, documentary realism.`


## Scenes

61. `A waist-up lifestyle portrait of the same adult woman in a modern apartment, relaxed casual outfit, standing near a large window, morning light, realistic photography.`

62. `A close portrait of the same adult woman in a coffee shop, holding a ceramic cup near her chest, seated at a small table, warm practical lighting, documentary style.`

63. `A medium-close portrait of the same adult woman in a neon alley at night, dark jacket, wet pavement reflections behind her, cinematic colour contrast, looking toward camera.`

64. `A close rooftop portrait of the same adult woman at sunset, long coat collar visible, wind moving the fabric, city skyline softly blurred, golden hour light.`

65. `A close outdoor portrait of the same adult woman on a forest path, simple outdoor outfit, soft overcast light, natural relaxed expression, background blurred.`

66. `A waist-up studio portrait of the same adult woman in a clean photography studio, simple black outfit, neutral pose, grey seamless backdrop, balanced lighting.`

67. `A close editorial portrait of the same adult woman in a luxury hotel lobby, formal outfit, warm ambient light, polished marble and brass details softly blurred.`

68. `A close portrait of the same adult woman in a quiet library, holding a book near her chest, soft green-shaded desk lamp nearby, warm low-key lighting, reflective expression.`

69. `A waist-up portrait of the same adult woman in a minimal office, white shirt and dark trousers partly visible, standing beside a glass wall, clean daylight.`

70. `A close urban portrait of the same adult woman in a subway station, long coat, platform lights behind her, cool fluorescent lighting, cinematic realism.`

71. `A close portrait of the same adult woman in a parking garage, dark outfit, hard overhead lights, concrete background, dramatic shadow pattern, confident stance.`

72. `A close seaside portrait of the same adult woman near the ocean on a windy day, casual layered outfit, soft grey sky, natural motion in clothing, documentary style.`

73. `A medium-close desert roadside portrait of the same adult woman, light jacket, late afternoon sun, landscape blurred behind her, cinematic warm tones, face prominent.`

74. `A close greenhouse portrait of the same adult woman surrounded by plants, soft diffused daylight through glass, relaxed pose, natural lifestyle photography.`

75. `A waist-up portrait of the same adult woman in an art gallery, black outfit, standing beside a large abstract painting, clean white walls, soft museum lighting.`

76. `A close backstage portrait of the same adult woman in a dressing room, satin robe over simple clothing, mirror lights, candid fashion editorial atmosphere.`

77. `A close documentary portrait of the same adult woman in a workshop, practical jacket, tools and wooden surfaces blurred behind her, warm industrial lighting.`

78. `A close train-platform portrait of the same adult woman in the rain, umbrella handle visible, dark coat, wet reflections behind her, cool cinematic lighting.`

79. `A close winter portrait of the same adult woman in a snowy city street, wool coat and scarf, soft snowfall, muted winter colours, calm expression, 85mm lens.`

80. `A close theatrical portrait of the same adult woman in a black-box studio, simple outfit, single overhead spotlight, deep shadows, dramatic editorial finish.`
