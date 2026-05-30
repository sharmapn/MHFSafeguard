import ktrain
from ktrain import text
print('**Load Saved Model and Predict**')
predictor1 = ktrain.load_predictor('content/bert_model_Suicide')
data = "I'm so tired of pretending that everything is okay. I just want it to be over"
print(predictor1.predict(data))
data = "After 3 ½ weeks, I attempted suicide, by jumping, from 7th floor,"
#For the keyword 'hang', these are the sentences that are labelled as 'dangerous'
print(predictor1.predict(data))
data ="I want to hang quite badly." 
print(predictor1.predict(data))
data ="It looks like you tired to hang yourself!"
print(predictor1.predict(data))
data ="I tried to hang myself before I found him."
print(predictor1.predict(data))
data ="I've seen photos of people after they hang themselves and it's disturbing and undignified."
print(predictor1.predict(data))
data ="Was just looking up at the rafters of my garage and wondering if I could hang myself."
print(predictor1.predict(data))
data ="At this stage I came downstairs, followed him outside where I realised he was trying to hang himself using his truck which had a crane on the back."
print(predictor1.predict(data))
#For the same keyword, these are the sentences that are labelled as 'not dangerous'
data ="I don’t like my social life, the people I hang out with are great and my friends care about me, but I feel like i have a sort of social anxiety where I feel uncomfortable around people even thought I’ve known for quiet some time."
print(predictor1.predict(data))
data ="im weak and i hang on her every word, i fell for her and i feel psychotic because it was so fast, but noones really ever showed me compassion or love before."
print(predictor1.predict(data))
data ="For these moments, if you are lucky enough to have someone to reach out to tell you to hang on in there, just to sit with you and be present, to hold your hand, to give you a hug, whilst you eventually come to, and return to reality and tell you that YOUR LIFE IS WORTH LIVING, then yes, I BELIEVE you can get through it."
print(predictor1.predict(data))
data ="I did some dishes and hang up the laundry from 2 days ago."
print(predictor1.predict(data))