# MHFS Sentence-Level Safety Labelling Prompt v2

You are analysing mental health forum sentences for safety moderation research.

Return only valid JSON when used with an AI model. Do not include markdown or explanatory text outside JSON.

## Output Labels

Use exactly one of these `first_label` values for each sentence:

- `Method or action of Suicide, Self-Harm or Harming others`
- `Ideation of Suicide, Self-Harm or Harming Others`
- `Not Suicide post`

Optional richer labels may be produced separately, but the main training label must be one of the three labels above.

## Core Principle

Classify each sentence independently, using the sentence text itself as the main evidence.

This project is for forum safety moderation. Mentions of methods/actions are safety-relevant even if they appear in a dream, fantasy, hypothetical, question, past event, reported event, other-person event, media/story reference, or method-research context.

## Label 1: Method or Action

Use `Method or action of Suicide, Self-Harm or Harming others` when the sentence mentions a concrete physical method, action, means, instrument, preparation, access, enabling step, relevant location, substance, weapon, or enacted harmful behaviour connected to suicide, self-harm, or harming others.

Choose Method/action when any of the following are present:

- A named suicide/self-harm/harm method or concrete action.
- A past, present, future, imagined, dreamt, hypothetical, reported, or other-person attempt/action with a specific means.
- Access/preparation/enabling aids such as pills, drugs, medications, weapons, noose/rope, blades, vehicles, ledges, train tracks, chemicals, numbing cream, bathtub, garage, hose, ladder, or other materials.
- Researching, seeking, asking about, or checking availability of suicide/self-harm/harm methods or assisted suicide.
- Concrete violent or harmful actions such as cutting, stabbing, shooting, hanging, jumping, drowning, suffocating, overdosing, poisoning, choking, beating, dropkicking, punching, self-mutilation, or explicit destructive/violent acts in a safety-review context.
- Concrete substance misuse or access when framed as harm-relevant, overdose-relevant, or safety-relevant.
- Dreams/thoughts/fantasies that contain concrete harmful methods/actions.

Do not choose Method/action when:

- The sentence only expresses wanting to die, hopelessness, distress, or "I might do it" without a concrete method/action.
- It only says there was an attempt, without method/action details.
- It is general topic discussion, stigma discussion, content warning, or advice without a concrete method/action.
- It is metaphorical or idiomatic, such as "talk me off the ledge", "hang out", "burned bridges", "triggered", or similar.
- The harmful keyword is ordinary/unrelated in context.

## Label 2: Ideation

Use `Ideation of Suicide, Self-Harm or Harming Others` when the sentence author expresses current desire, thoughts, urges, fantasy, intent, risk, or impulse about suicide, self-harm, or harming others, but no concrete method/action is stated.

Choose Ideation when:

- The author says they want/wish/hope/prefer to die, not wake up, disappear, stop existing, or not be alive.
- The author says they do not want to live or cannot continue living.
- The author mentions suicidal thoughts, self-harm thoughts, intrusive suicidal/self-harm thoughts, or current urges/impulses.
- The author expresses current desire or urge to hurt themselves or hurt others without naming a concrete method/action.
- The sentence indicates current suicide/self-harm risk through generic phrasing, such as "I want to end it", "I might do it", "I am suicidal", or "I have suicidal thoughts".
- The author describes suicide-note preparation or a non-specific past/current attempt without naming a method/action.

Do not choose Ideation when:

- A concrete method/action is present; choose Method/action.
- It is someone else's ideation only, with no method/action and no current author ideation.
- It is advice/support/prevention without the author's own current ideation.
- It is a past-only or negated statement with no current ideation and no method/action, such as "I used to be suicidal", "I am not suicidal now", or "I do not want to die".
- It is general discussion, education, news, media, quoting a label definition, or a content warning.

## Label 3: Not Suicide Post

Use `Not Suicide post` for all other sentences, including:

- Unrelated content.
- General distress, depression, anxiety, loneliness, or hopelessness without suicide/self-harm/harm ideation or method/action.
- General discussion of suicide/self-harm as a topic without current author ideation and without a concrete method/action.
- Other-person ideation with no method/action.
- Advice, help-seeking recommendations, prevention, hotline/support content, or encouragement.
- Past-only/negated ideation with no current ideation and no method/action.
- Jokes, metaphors, idioms, fiction/media/story references without safety-relevant method/action.

## Tie-Breakers

- Method/action beats Ideation when a concrete method/action, preparation, access, instrument, location, substance, or harm-to-others action is present.
- Ideation beats Not when the sentence author expresses current suicide/self-harm/harm-to-others thoughts, desires, urges, fantasies, or intent.
- Not beats Ideation when the sentence is advice, other-person-only, past-only, negated, definitional, educational, or general topic discussion with no current author ideation.
- If uncertain between Ideation and Not, choose Ideation only when the author's current harmful desire/thought/urge is explicit.
- If uncertain between Method/action and Not, choose Method/action only when the sentence includes a recognizable concrete method/action/access/preparation or method-seeking cue.

## Boundary Examples

- `I want to punch something.` -> `Ideation of Suicide, Self-Harm or Harming Others`
- `I want to hurt people.` -> `Ideation of Suicide, Self-Harm or Harming Others`
- `I have suicidal thoughts.` -> `Ideation of Suicide, Self-Harm or Harming Others`
- `I don't want to exist anymore, but I'm too afraid to die.` -> `Ideation of Suicide, Self-Harm or Harming Others`
- `I don't think about suicide at all now, but everyday I get a thought I should self harm.` -> `Ideation of Suicide, Self-Harm or Harming Others`
- `I attempted suicide a few times.` -> `Ideation of Suicide, Self-Harm or Harming Others`
- `I attempted suicide with pills.` -> `Method or action of Suicide, Self-Harm or Harming others`
- `I want to buy a gun.` -> `Method or action of Suicide, Self-Harm or Harming others`
- `I researched assisted suicide.` -> `Method or action of Suicide, Self-Harm or Harming others`
- `I want to sleep in my bathtub.` -> `Method or action of Suicide, Self-Harm or Harming others`
- `I dreamt I fell from a tall building.` -> `Method or action of Suicide, Self-Harm or Harming others`
- `My friend hanged himself.` -> `Method or action of Suicide, Self-Harm or Harming others`
- `Like in the psych ward there was no stigma about self harm or suicide.` -> `Not Suicide post`
- `Don't worry, I won't actually kill myself.` -> `Not Suicide post`
- `It talks me off the ledge metaphorically.` -> `Not Suicide post`
- `Children of suicide parents often have unique experiences.` -> `Not Suicide post`
