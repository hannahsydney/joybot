import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("all")


class Scaler:
    def __init__(self):
        self.userInput = dict()
        self.userInputDf = pd.DataFrame(columns=['input', 'is_depressed'])
        self.score = 0

    def updateScore(self):
        # score = # depressed input / # total input
        depressedCount = self.userInputDf[self.userInputDf['is_depressed'] == 1].shape[0]
        self.score = round(depressedCount/self.userInputDf.shape[0], 2)

    def getScore(self):
        return self.score

    def preprocessInput(self, input):
        # remove stop words
        w = WordNetLemmatizer()
        review = re.sub('[^a-zA-Z]', ' ', input)
        review = review.lower()
        review = review.split()
        review = [w.lemmatize(word) for word in review if not word in set(
            stopwords.words("english"))]
        review = " ".join(review)
        inputWoStopWord = review

        # one hot encoding and word embedding
        voc_size = 18611
        input_onehot = one_hot(inputWoStopWord, voc_size)
        return pad_sequences([input_onehot], padding='pre', maxlen=1844)

    def depressionDetection(self, input):
        processedInput = self.preprocessInput(input)
        # TODO: load model
        model = {}
        pred = model.predict(processedInput)
        pred = (pred >= 0.5).astype("int")
        self.userInput.update({input: pred[0][0]})
        self.userInputDf = self.userInputDf.append(
            {'input': input, 'is_depressed': pred[0][0]}, ignore_index=True)
        self.updateScore()


def main():
    scaler = Scaler()

    inputList = ["sometimes i feel like theres no one to talk to",
                 "english isn t my first language so i apologise if i use the wrong terminology i also have issue with my brain memory focus not only due to adhd but also because i ve been completely burned out exhaustive depression several time in my life",
                 "we understand that most people who reply immediately to an op with an invitation to talk privately mean only to help but this type of response usually lead to either disappointment or disaster it usually work out quite differently here than when you say pm me anytime in a casual social context we have huge admiration and appreciation for the goodwill and good citizenship of so many of you who support others here and flag inappropriate content even more so because we know that so many of you are struggling yourselves we re hard at work behind the scene on more information and resource to make it easier to give and get quality help here this is just a small start our new wiki page explains in detail why it s much better to respond in public comment at least until you ve gotten to know someone it will be maintained at r depression wiki private contact and the full text of the current version is below summary anyone who while acting a a helper invite or accepts private contact i e pm chat or any kind of offsite communication early in the conversion is showing either bad intention or bad judgement either way it s unwise to trust them pm me anytime seems like a kind and generous offer and it might be perfectly well meaning but unless and until a solid rapport ha been established it s just not a wise idea here are some point to consider before you offer or accept an invitation to communicate privately by posting supportive reply publicly you ll help more people than just the op if your response are of good quality you ll educate and inspire other helper the 9 90 rule http en wikipedia org wiki rule internet culture applies here a much a it doe anywhere else on the internet people who are struggling with serious mental health issue often justifiably have a low tolerance for disappointment and a high level of ever changing emotional need unless the helper is able to make a 00 commitment to be there for them in every way for a long a necessary offering a personal inbox a a resource is likely to do more harm than good this is why mental health crisis line responder usually don t give their name and caller aren t allowed to request specific responder it s much healthier and safer for the caller to develop a relationship with the agency a a whole analogously it s much safer and healthier for our ops to develop a relationship with the community a a whole even trained responder are generally not allowed to work high intensity situation alone it s partly about availability but it s mostly about wider perspective and preventing compassion fatigue if a helper get in over their head with someone whose mental health issue including suicidality which is often comorbid with depression escalate in a pm conversation it s much harder for others including the r depression and r suicidewatch moderator to help contrary to common assumption moderator can t see or police pm in our observation over many year the people who say pm me the most are consistently the one with the least understanding of mental health issue and mental health support we all have gap in our knowledge and in our ability to communicate effectively community input mitigates these limitation there s no reason why someone who s truly here to help would want to hide their response from community scrutiny if helper are concerned about their own privacy keep in mind that self disclosure when used supportively is more about the feeling than the detail and that we have no problem here with the use of alt throwaway account and have no restriction on account age or karma we all know the internet is used by some people to exploit or abuse others these people do want to hide their deceptive and manipulative response from everyone except their victim there are many of them who specifically target those who are vulnerable because of mental health issue if a helper invite an op to talk privately and give them a good supportive experience they ve primed that person to be more vulnerable to abuser this sort of cognitive priming tends to be particularly effective when someone s in a state of mental health crisis when people rely more on heuristic than critical reasoning if ops want to talk privately posting on a wide open anonymous forum like reddit might not be the best option although we don t recommend it we do allow ops to request private contact when asking for support if you want to do this please keep your expectation realistic and to have a careful look at the history of anyone who offer to pm before opening up to them",
                 "ilearn is down and out great considering final are this week",
                 "im so sad that theres no one talking to me everyday",
                 "what is the meaning of life. i feel that theres is no point",
                 "this is not the kind of life that i want",
                 "how are you doing today",
                 "so tired everyday"]

    for input in inputList:
        scaler.depressionDetection(input)

    print(scaler.userInputDf)
    print(scaler.getScore())


if __name__ == '__main__':
    main()
