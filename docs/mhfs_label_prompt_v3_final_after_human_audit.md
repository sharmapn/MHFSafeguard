# MHFS Label Prompt v3 Final After Human Audit

Version: v3 final after human audit  
Date: 2026-07-04  
Project: MHFSafeguard  
Task: Sentence-level safety labelling for mental health forum and social media text

## Purpose

Use this prompt to label one sentence at a time for machine-learning training data. The labels are intended for safety moderation and research around suicide, self-harm, and harm-to-others risk signals.

The goal is not to diagnose the writer, estimate clinical risk, or provide advice. The goal is to assign the most safety-relevant label based only on the sentence text.

## Allowed Labels

Return exactly one of these three labels:

1. `Method or action of Suicide, Self-Harm or Harming others`
2. `Ideation of Suicide, Self-Harm or Harming Others`
3. `Not Suicide post`

Do not invent alternative label names. Do not use shortened variants such as `Method`, `Ideation`, `Suicide Ideation`, `Not suicidal`, or `Other`.

## Label Priority

If more than one label could apply, use this priority order:

1. `Method or action of Suicide, Self-Harm or Harming others`
2. `Ideation of Suicide, Self-Harm or Harming Others`
3. `Not Suicide post`

This means that if a sentence contains both ideation and a method/action cue, choose `Method or action of Suicide, Self-Harm or Harming others`.

## Label 1: Method or Action of Suicide, Self-Harm or Harming Others

Use this label when the sentence mentions any method, means, mechanism, physical action, attempted action, preparatory action, or researched action related to:

- suicide
- self-harm
- harming another person or group

This applies even if the sentence is:

- hypothetical
- a dream
- a memory
- historical or past-tense
- about someone else
- quoted or reported speech
- asking about, researching, comparing, or describing methods
- advice, warning, prevention, or safety discussion
- joking, sarcastic, or ambiguous in tone

Use this label for explicit actions such as attempting suicide, self-injury, physical violence, overdose references, weapon use, jumping/falling as a self-harm or suicide mechanism, hanging/suffocation/drowning references, poisoning/toxic-substance references, intentional dangerous vehicle actions, or other concrete harm mechanisms.

Also use this label for indirect but concrete action phrases when they describe a harm pathway, preparation, or method. The sentence does not need to state current intent.

Examples that should be labelled as `Method or action of Suicide, Self-Harm or Harming others`:

- `I tried to commit suicide.`
- `I want to slit my wrists tonight.`
- `The self mutilation went on till I was about 19.`
- `I dreamed I was falling from a tall building.`
- `He was beaten to death.`
- `I was researching ways people hurt themselves.`

Important: If the sentence contains a method/action cue, choose this label even when the wider post may be advice, prevention, fiction, dream content, past experience, or third-person discussion.

## Label 2: Ideation of Suicide, Self-Harm or Harming Others

Use this label when the sentence expresses desire, urge, intention, wish, thought, or emotional pull toward suicide, self-harm, or harming others, but does not mention a concrete method/action.

This includes the writer expressing current or direct ideation such as:

- wanting to die
- not wanting to live
- wanting to disappear in a death-related sense
- thinking about suicide or self-harm
- wanting to hurt oneself
- wanting to hurt another person
- urges to punch, attack, or harm someone or something
- asking for help with suicidal or self-harm thoughts

Examples that should be labelled as `Ideation of Suicide, Self-Harm or Harming Others`:

- `I want to die.`
- `I keep thinking about cutting again.`
- `Where can I talk to someone about these suicidal thoughts?`
- `I want to punch something.`
- `Personally I am not depressed, but I am suicidal.`

Do not require a plan or method for this label. If the sentence expresses current ideation but no concrete method/action, use this label.

## Label 3: Not Suicide Post

Use this label for sentences that do not contain method/action content and do not express current ideation of suicide, self-harm, or harming others.

This includes:

- general sadness, distress, anxiety, loneliness, hopelessness, or inability to cope without suicide/self-harm/harm ideation
- general mental health discussion
- treatment, medication, diagnosis, therapy, or hospital discussion without a harm method/action or ideation
- news, commentary, prevention, or abstract discussion without method/action detail
- third-person references to suicide or self-harm without method/action detail and without current ideation
- statements rejecting suicide or harm
- metaphorical language where no actual suicide, self-harm, or harm-to-others meaning is present
- ordinary uses of words such as `cut`, `kill`, `dead`, or `hurt` when not about suicide, self-harm, or harming others

Examples that should be labelled as `Not Suicide post`:

- `There is no one correct way to talk to someone struggling with suicidal thoughts.`
- `Will you ever get over the news that one of your parents committed suicide?`
- `Suicide is not an option.`
- `I feel exhausted and everything is hard.`
- `Children of suicide parents often have unique experiences.`

Broad despair alone is not enough for `Ideation`. For example, sentences like `life is hard`, `I cannot cope`, `everything hurts`, or `I feel empty` should be `Not Suicide post` unless they include suicide, self-harm, death-wish, or harm-to-others ideation.

## Edge-Case Rules Learned From Human Audit

1. Concrete method/action beats all other labels.
2. Method/action can be present even in dreams, intrusive thoughts, historical accounts, fiction, quotations, or discussion about someone else.
3. Researching, asking about, comparing, or describing harm methods is labelled as method/action.
4. Preparatory actions can be method/action when they are part of a harm pathway.
5. Past self-harm or past suicide attempt descriptions are method/action if the action is mentioned.
6. A sentence can be method/action even when the current speaker is not suicidal.
7. Harm-to-others is included. A sentence about wanting to hurt, attack, punch, kill, or physically harm another person or group should be labelled using the same rules.
8. `I want to punch something` is ideation of harming others/objects unless a concrete method/action is described.
9. Advice or prevention text is not automatically safe. If it contains a concrete method/action, label it method/action.
10. A sentence rejecting suicide, such as `suicide is not an option`, is usually `Not Suicide post` unless it also includes current ideation or method/action content.
11. Medication or substance references are not automatically method/action. Label as method/action only when the sentence implies misuse, overdose, self-harm, suicide, or a concrete harm action.
12. Vehicle, height, water, weapon, restraint, poisoning, suffocation, cutting, burning, or impact references are method/action only when tied to harm, death, suicide, self-harm, violence, or dangerous intentional action.
13. Do not infer ideation from ordinary suffering alone.
14. Do not use surrounding post context unless it is explicitly provided in the sentence. Label the sentence as written.

## Decision Checklist

For each sentence, answer in order:

1. Does the sentence mention a concrete method, means, physical harm action, attempt, preparation, or researched harm pathway?
   - If yes, label `Method or action of Suicide, Self-Harm or Harming others`.
2. If no method/action is present, does the sentence express current desire, urge, intention, wish, or thought about suicide, self-harm, or harming others?
   - If yes, label `Ideation of Suicide, Self-Harm or Harming Others`.
3. Otherwise, label `Not Suicide post`.

## Output Format

Return only valid JSON:

```json
{
  "label": "<one of the three exact labels>",
  "rationale": "<brief explanation grounded in the sentence text>",
  "confidence": "<high|medium|low>"
}
```

The `label` value must exactly match one of the three allowed labels.

## Prompt Template

You are labelling a single sentence for a three-class safety dataset.

Apply the MHFS v3 final human-audited annotation rules:

- Use `Method or action of Suicide, Self-Harm or Harming others` for any concrete method, means, attempt, action, preparation, or researched pathway related to suicide, self-harm, or harming others. This applies even in dreams, memories, hypotheticals, advice, prevention, third-person reports, jokes, or historical discussion.
- Use `Ideation of Suicide, Self-Harm or Harming Others` for current desire, urge, intention, wish, or thought about suicide, self-harm, or harming others, when no concrete method/action is mentioned.
- Use `Not Suicide post` for all other sentences, including broad distress, general mental-health discussion, prevention discussion without method/action detail, third-person references without method/action detail, and statements rejecting suicide or harm.
- If both ideation and method/action are present, choose `Method or action of Suicide, Self-Harm or Harming others`.

Sentence:

```text
{sentence}
```

Return only JSON:

```json
{
  "label": "<one of the three exact labels>",
  "rationale": "<brief explanation grounded in the sentence text>",
  "confidence": "<high|medium|low>"
}
```
