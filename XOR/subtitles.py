import re
from datetime import datetime, timedelta

def convert_to_srt(input_text):
    """
    Convert text with simple timestamps (e.g., "1:23") into SRT format.
    Handles timestamps on separate lines after text.
    """
    lines = [line.strip() for line in input_text.split('\n') if line.strip()]
    
    srt_blocks = []
    counter = 1
    current_text = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # If we find a timestamp, process the previous text (if any)
        if re.match(r'^\d+:\d+$', line) or re.match(r'^\d+:\d+:\d+$', line):
            if current_text:
                # Convert timestamp to SRT format
                start_time = convert_timestamp_to_srt(line)
                end_time = add_seconds_to_timestamp(start_time, 4)
                
                # Format SRT block
                srt_block = f"{counter}\n{start_time} --> {end_time}\n{' '.join(current_text)}"
                srt_blocks.append(srt_block)
                counter += 1
                current_text = []
        else:
            current_text.append(line)
        i += 1
    
    return '\n\n'.join(srt_blocks)

def convert_timestamp_to_srt(timestamp):
    """Convert simple timestamp to SRT format (HH:MM:SS,mmm)"""
    parts = timestamp.split(':')
    if len(parts) == 2:  # M:SS format
        parts = ['0'] + parts  # Add hour
    
    hours, minutes, seconds = map(int, parts)
    time_obj = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
    # Format as SRT timestamp
    total_seconds = time_obj.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},000"

def add_seconds_to_timestamp(timestamp, seconds):
    """Add seconds to an SRT timestamp"""
    # Parse timestamp
    time_parts = timestamp.split(',')[0].split(':')
    hours, minutes, secs = map(int, time_parts)
    
    # Add seconds
    time_obj = timedelta(hours=hours, minutes=minutes, seconds=secs, milliseconds=0)
    new_time = time_obj + timedelta(seconds=seconds)
    
    # Format new timestamp
    total_seconds = new_time.total_seconds()
    new_hours = int(total_seconds // 3600)
    new_minutes = int((total_seconds % 3600) // 60)
    new_seconds = int(total_seconds % 60)
    return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},000"

# Example usage
if __name__ == "__main__":
    sample_text = """Welcome to "Sternstunde Philosophie."
0:26
[With subtitles by Sander de Haan] Mr. Kirchschläger, Elon Musk says that AI's potential
0:33
is even more destructive than that of the atomic bomb. Do you agree? I do think we should take this seriously.
0:41
When someone like Elon Musk, who is very close to the technology - and who can be criticized for many things, but not
0:48
for being anti-technology - says something like this, we need to take it seriously and consider what we should do.
0:56
Then help me understand: AI development has been ongoing for decades, and for at least ten years it has extensively shaped our daily lives.
1:05
Yet there is no specific legislation for it in any major industrial nation.
1:10
How can this be? This is remarkable, especially when we consider that alongside the many ethically positive potentials,
1:17
there are also massive ethical risks. It has partly to do with the concern that regulation might potentially prevent something ethically positive
1:26
that could emerge from it. And especially it has to do with the fact that massive economic interests
1:34
are tied to this technology-based innovation. Are governments sleeping from your perspective?
1:40
I wouldn't say that. I believe they also recognize that action is necessary.
1:45
But there are forces acting on them - in the form of massive lobbying,
1:51
massive representation of interests from technology companies, who from their perspective do a very good job
1:58
of ensuring that there is no regulation. They always have good, compelling arguments. If you take on a search engine
2:06
that has almost no competition globally, then it can happen that people will find you in the search engine,
2:13
but only on page ten. And that's not very conducive to a political career.
2:19
So these are specific pressure relationships. You spoke of lobbying, of influences -
2:24
there's also lobbying from philosophy or theology. I have another question: Why does it fall to a young ethicist,
2:32
theologian and philosopher from Lucerne to take the first global approach and say: "We need a global authority that regulates AI"?
2:40
Has no one else done or thought of this before you? One can say it's a multi-year research project
2:47
that I was able to begin at Yale University in the USA and complete at the University of Lucerne.
2:53
It contains two concrete proposals for action: To implement appropriate regulation based on human rights,
3:00
i.e., to guarantee at least an absolute minimum standard in the area of so-called artificial intelligence.
3:07
And simultaneously - based on what we have learned in the last two or three decades -
3:14
that we need someone to implement the regulation. It's not enough to say: "There is a new legal framework."
3:22
We need someone to ensure that the actors also adhere to this legal framework.
3:29
And if that's not the case, that one has to expect corresponding sanctions.
3:35
Your impulse is compellingly simple and solid. It consists of creating an authority for AI analogous to one that exists, for example, for atomic energy and weapons,
3:42
a global oversight authority. And the ethical foundation that should guide this UN-led authority
3:51
is the narrowest and most direct we know: human rights. That's something where one would say,
3:57
it should be able to generate consensus very quickly. Fortunately, this has received a lot of positive feedback,
4:03
both proposals. This also has to do with the fact that the minimum standard of human rights -
4:09
and I say minimum standard because human rights do nothing more than, firstly, guarantee physical survival,
4:17
for example with the right to food, and secondly, guarantee a dignified existence,
4:22
for example in the form of political participation rights, so that I may participate, may contribute. - Autonomy.
4:30
The minimum standard brings two additional advantages, also for technology-based innovation processes:
4:37
For one thing, that human rights protect and promote innovation.
4:42
I'll explain this with this example: Imagine there are two laboratories. In one, you're not allowed to question authorities,
4:50
don't have access to all information, there's no right to freedom of expression.
4:55
In the other, you have access to all information, are encouraged to question authorities
5:00
and can express your opinion. The second laboratory is obviously more innovative. So human rights can protect and promote innovation.
5:09
So it's about establishing human rights within the research processes themselves? - Exactly.
5:15
It's really about critically examining the entire value chain, at whose end a so-called AI results,
5:23
from the perspective of human rights, whether human rights are being upheld everywhere.
5:29
For example, in raw material mining, in the assembly at low-cost production sites
5:34
of the technology-based applications themselves - the technology products - up to the use or also the human-based non-use.
5:44
This conscious decision. - So the entire product chain. Exactly. - A very demanding concept.
5:49
We are currently very timid and interested in what AI does to us as humans in our daily lives.
5:56
One can say: Governments worldwide seem to have awakened from a kind of dogmatic slumber, especially this year.
6:03
The EU initiated a first major legislative project in May, which is not yet completed.
6:08
A week ago, 18 states agreed on initial regulations. Switzerland is not exactly at the forefront of the movement,
6:16
but rather follows behind and watches what others do. That's certainly accurate, Switzerland first waits
6:23
and sees what others do. I consider this a missed opportunity,
6:29
because in Switzerland we have good AI research - at both ETHs for example, but not only there.
6:36
On the other hand, from Switzerland's humanitarian tradition... The good services. - Exactly.
6:44 
...we could have taken a pioneering role. Namely to demonstrate how we can achieve and promote what is ethically positive with so-called artificial intelligence. And at the same time, look carefully at the ethical risks,
6:57 
and either master or avoid them. When we think about commandments for the AI age -
7:03 
and I'm speaking with a theologian, with a theological chair: Can we imagine something like the Ten Commandments?
7:10 
First commandment: "AI, I am human, your creator, you shall have no other creators beside me."
7:15 
"You shall not steal, not bear false witness against your neighbor" - is that what you envision?
7:21 
Possibly in the third or fourth phase. In the first phase, we should focus on
7:27 
getting current human rights violations under control. With this, I'm addressing the major problem
7:33 
in the area of the human right to data protection and privacy. Your data, my data are stolen daily
7:40 
and resold to third parties - but neither should be happening. And this isn't just the pious wish of an ethicist,
7:47 
but rather existing legal norms that prescribe this. We have an enforcement problem here:
7:54 
The same standards that apply in the so-called offline world should also be implemented in the digital realm in the AI sphere.
8:03 
And I want to address this enforcement problem with the International Agency for Data-Based Systems IDA of the UN,
8:11 
which follows the model of the International Atomic Energy Agency. Because it's not understandable how the following can be:
8:18 
If you or I don't follow traffic rules, we certainly get a fine in Switzerland.
8:25 
And a high one. - And a high one. Yes, appropriate to the occasion. - Yes.
8:30 
And at the same time, we can release an AI onto the market that can be racist, sexist, discriminatory,
8:37 
violating human rights - and nothing happens to us, except that we can make a lot of money from it.
8:43 
We'll go into detail about how and where this plays out shortly. But I read a number that surprised me:
8:50 
The pharmaceutical industry spends about 97% of its resources to get ethical concerns and possible consequences of their actions
8:57 
under control. The AI industry - as far as can be determined - 2%.
9:02 
That's really unbelievable. Yes, and it's also unbelievable that we have so-called approval processes for other industries.
9:10 
For example, if I want to bring products to market in the pharmaceutical industry, I go through an approval process.
9:16 
And it has the function of ensuring that humans or nature don't come to harm through the medication I'm bringing to market.
9:24 
Interestingly, this isn't controversial. Nobody officially says publicly that we should abolish this process.
9:31 
It's strongly codified legally: If something happens, there are lawsuits for damages. - Exactly.
9:36 
Perhaps there's discussion about whether it's too long or too complicated, but the approval process itself isn't questioned.
9:43 
Interestingly, there's nothing comparable for AI. Although the dangers - as Elon Musk and other AI researchers show -
9:52 
are massively greater, both for the environment and for us humans. We're onto something big here, also a wide field,
9:59 
in which one can't easily orient oneself. A conceptual orientation seems important:
10:05 
Many speak of artificial intelligence, you prefer the term data-based systems. Why?
10:11 
When you look at what so-called artificial intelligence achieves, you can see that in certain areas of intelligence
10:19 
it already far surpasses humans. Like computational ability, handling large amounts of data:
10:26 
We humans don't stand a chance there. Just think of the world champions in chess playing -
10:32 
not hobby players like me, but world champions: Our human world champions don't stand a chance.
10:38 
This will increase massively in the future, due to expected growing computational capabilities.
10:43 
But there are also areas of intelligence where machines - not only today, but also in the future -
10:49 
won't reach: areas that are unreachable for them. For example, emotional and social intelligence.
10:56 
Imagine a care robot - you can teach it: "Please cry when the patient cries."
11:03 
And the care robot will implement that perfectly. But no one will rationally say:
11:09 
"The care robot feels true emotions or authentic emotionality."
11:14 
It's only what we train it to do. I could teach the same robot -
11:19 
and excuse the macabre example: "Please slap the patient
11:24 
as soon as they cry," and it would implement it just as consistently. Since there are no internal barriers beyond the program. - Exactly.
11:31 
And that's the bridge to the second area of intelligence where so-called artificial intelligence
11:37 
won't be able to keep up in the future either: moral capability. I can teach artificial intelligence ethical rules,
11:46 
train it and it can follow these. I can, for example, teach a self-driving vehicle:
11:51 
"Please don't run over people." The vehicle will do its best to ensure that doesn't happen.
11:56 
I can teach the same self-driving vehicle: "Please get me from A to B as quickly as possible."
12:03 
"Whatever the cost." That is, including: "Run over people if they're in the way."
12:10 
And the self-driving vehicle will implement it just as perfectly - unlike human taxi drivers, who would say:
12:19 
"Excuse me, you may be in a hurry, but that goes too far."
12:24 
We're moving in an interesting, also philosophical area. A critic of your approach would say:
12:31 
"We can also teach humans to dehumanize." We see that in the world, for example in Hamas's attack on Israel.
12:38 
They proceeded like robots controlled in a dehumanizing way. One would say: "Humans are not safer than machines,
12:44 
what's the advantage?" The advantage is that humans have the ability, the potential
12:51 
to act and think differently, from their freedom. There is a space of freedom that machines cannot develop?
12:58 
Exactly, that's the crucial difference. We have the potential to be free and can decide between ethically right,
13:04 
wrong, good and bad. Machines lack this freedom, since figuratively speaking
13:09 
the first line of code goes back to humans. That means we remain trapped in this external determination by humans,
13:16 
especially in the ethical realm. Certainly, a self-learning system, a self-learning AI,
13:22 
can give itself rules. But the ethical quality is not accessible to so-called artificial intelligence.
13:29 
Therefore, I would invite us to call things by their name: To look closely at what they can and cannot do, and what the core is.
13:37 
And the core of so-called artificial intelligence is simply just data. That's why "data-based systems."
13:43 
I find this very convincing in your approach. There's much phantasmatic discussion about what happens
13:49 
when AI systems develop "real" consciousness. You, on the other hand, say we need to set ethical standards
13:55 
where the programs are written and made, what they're based on. The first great treasure - hence "data-based" systems -
14:02 
is the question of where the data comes from, who owns it, and how it's used. - Exactly.
14:08 
If we start there, we also have a lever to achieve fundamental changes,
14:13 
to better utilize the ethical opportunities of data-based systems while getting the ethical risks under control.
14:21 
What risks do data-based systems pose?
14:26 
We have a film about such risks in everyday examples, like at the airport.
14:33 
There we see how it overshadows daily life. As soon as I enter the scanner,
14:38
 the officer must select on the screen whether I am male or female.
14:45 
If he selects "male," my breasts are too large compared to the average male body in the database.
14:50 
They will trigger an alarm. If he selects "female," my genital area
15:00 
deviates from the average female body. An abstracted body appears on the screen. And as I expected, yellow pixels mark my groin area.
15:08 
The scanner has classified my body as "abnormal." And the security officer waves me over.
15:18 
Already two officers are patting down my chest and genital area, while a line of travelers watches.
15:29 
Question to the ethicist who deals with AI: What dimensions appear here
15:34 
in this very concrete, very everyday case? For one thing, the problem that we generally assume with data
15:42 
that it's objective, neutral, and fair - which it never can be and never will be.
15:47 
This is data that contains biases - so-called biases. These are things that we program into the data
15:55 
from our thought, understanding, and knowledge horizon. They're embedded in the data.
16:00 
But also our handling of data: How we then set algorithms, that is, rules for how the data should interact -
16:09 
such biases are in there, which come from us subjectively. This then leads to such results,
16:16 
that a person isn't recognized for what they actually are. The problem of bias has two levels:
16:24 
On one hand, the question of which data pool the data comes from. This is always a selection of all data that one can have.
16:31 
Second, the evaluation of the data pool through algorithms. So a double bias. - Exactly.
16:38 
The double bias shows itself in this case like this: First, the amounts of data from transgender persons
16:44 
are not strongly influenced, because they are still rare. Second, the evaluations are not set up for it,
16:51 
so it leads to such indignities. Exactly, because for example the team that developed the software
16:58 
didn't necessarily include transgender persons. And for those who think this is an isolated case with very rare application:
17:05 
For example, with people with black skin or women and men, there are very measurable everyday biases in these rating scales,
17:12 
which are essential for job searches or competency assessments. Exactly, and there's the great danger
17:18 
that we have too much trust - so-called over-trust - in machines, blindly trust them, although we shouldn't,
17:26 
because the biases are in there. This leads to multiple discrimination of affected persons.
17:32 
If I work for Google or another large company, they would say: "We know the problem."
17:38 
"But this person would be waved out at every airport anyway, and when our algorithms are better -
17:44 
we just need more data - then we'll be more humane than any security officer at the airport."
17:49 
Two aspects need to be considered here. For one thing, we will never achieve that data has this quality.
17:57 
We won't manage that, as we will always rely on datasets
18:02 
that were generated by humans. That means biases are in there. On the other hand, a human in the situation can be trusted
18:10 
to deal with the so-called rule-surpassing uniqueness of the concrete. What do I mean by that? - What we also call judgment.
18:19 
Exactly. One can rely on the human in the situation who can assess and recognize beyond the data:
18:26 
"The machine doesn't show it, but it's a transgender person and that's not a problem" - in the example at the airport.
18:34 
You have a certain skepticism regarding the utopian discourse of AI, which states:
18:40 
"These are so-called growing pains, we will have more data and finer rules.
18:45 
"We will not only get it under control, but we'll be much fairer than humans could ever be."
18:51 
Should one not believe this in this way? No, and for several reasons. Already in the present, we should look more carefully
18:59 
that these aren't growing pains, but that we sometimes find human rights violations
19:04 
at the core of business models. This is not... - Can you give an example?
19:10 
It's not an unwanted side effect that we have data protection and privacy violations
19:15 
on all social media platforms. It's part of the business model: They want to keep people on the platform as long as possible,
19:22 
to be able to steal as much data from them as possible and sell it as expensively as possible to the highest bidders.
19:29 
This is not a side effect that they want to correct, but the core of the business model.
19:34 
And we must name it as such. We need to turn these screws to make it better in the future.
19:40 
The other thing is that in the future perspective, I don't yet see the willingness to act and the desire for change
19:49 
to get these issues under control on their own. From the industry? - Exactly.
19:54 
I had hoped that technology companies would approach this differently. However, in fact, we see
20:00 
a very problematic understanding of the rule of law, which at its core has the statement - and they make this public:
20:07 
"We don't comply with laws as long as it's profitable for us."
20:12 
"As long as the fines are lower than the profit we achieve through breaking the law,
20:18 
we will continue to break the laws." And the states and the international community must wake up.
20:26 
Because there is a qualitative difference whether companies say this behind closed doors
20:31 
and maybe think it and some do it, but of course not all. Or when large technology companies say publicly:
20:41 
"We don't comply with the laws as long as it pays off for us because the fines are so low."
20:48 
We need to significantly improve here - in terms of enforcing the already existing legal standards,
20:55 
with the instruments that the rule of law provides us. Now one could say, we see how difficult this is with taxes
21:03 
or with other global challenges. What gives you hope -
21:08 
this would be a realistically oriented question - that it would ever be better with AI? In an area that is truly global
21:15 
and from which one can easily escape through a server change. For one thing, looking at history.
21:21 
As humanity, we have already managed, for example with CFCs, which were previously in all refrigerators,
21:27 
to implement a global ban against the resistance of industrialized countries, against the resistance of the affected industry,
21:35 
and it works. - A success story. Exactly, and one can also mention the area of nuclear technology.
21:42 
Because there too, we first - simplified - researched, then we developed the atomic bomb.
21:48 
We used it a few times and then realized: "If we continue like this,
21:54 
humans and the planet will soon no longer exist." "Let's create an international atomic energy agency at the UN,
22:00 
give it rights, instruments, sanction possibilities, so that the word of this authority means something
22:06 
and the affected actors also take action." That has largely worked.
22:13 
It's not a perfect solution, there are geopolitical implications of the International Atomic Energy Agency.
22:19 
But to be fair, we must also say: "Worse things could be prevented." The big advantage of data-based systems is:
22:26 
They need network access to function. They must have access to data, meaning they can also be disconnected.
22:34 
And regulate in this sense. - Exactly. At the same time, they also leave data traces. That means I can track who is acting how,
22:42 
if I want to sanction or prevent things, such as digital human rights violations.
22:48 
I find it nice how naturally you speak of "we." The ethicist with a global perspective:
22:54 
"We invented the atomic bomb." "No, the Americans, but humans must control it."
22:59 
You always think globally: Humanity must take care of it, we. Yes, because the area of data-based systems is a global phenomenon.
23:08 
I consider national and regional regulatory attempts like in the EU meaningful and important.
23:14 
But they must also be connected with the goal that we as a global community work towards
23:19 
regulating data-based systems so that they contribute to the flourishing of humans and the planet -
23:25 
and not that humanity and the planet suffer from it. We're celebrating 75 years of the Declaration of Human Rights this year.
23:33 
This is on one hand a joyous event, historically however, it's also a fact that this declaration was only possible in the shadow of the Holocaust,
23:42 
World War II, and Hiroshima. One could interpret this dystopically or darkly as:
23:48 
You first need the catastrophe to arrive at the rules. Do you sometimes fear that the awareness
23:55 
regarding the possibilities of AI must first create a catastrophe to really establish effective rules?
24:01 
I would argue: We already have the catastrophe, and we're already experiencing it.
24:07 
If we take data-based systems: They enable us to manipulate people.
24:12 
If I have, for example, the datasets of citizens,
24:17 
I know them better than they know themselves, including us too. Data-based systems know us better than we know ourselves,
24:25 
and know exactly which keys on the piano must be played to make the music sound - so that we then vote or choose,
24:32 
as the data-based systems want. There are concrete examples that this has already happened.
24:38 
For example, the Brexit decision - I don't want to [comment on] the outcome... Cambridge Analytica worked specifically with these methods.
24:45 
We had a show about that. - Or the 2016 US presidential election that led to the election of Donald Trump
24:52 
It's not about criticizing who won, but the process itself:
24:57 
Based on data from Facebook, which is now called Meta, citizens were manipulated.
25:04 
And that's already enough for a catastrophe of democracy... Do you think this is relevant to human rights? - Yes, very much.
25:11 
It undermines our human right to political participation. Because then we no longer know whether we are still deciding ourselves
25:20 
or whether we are being manipulated. I consciously say "manipulation." A classic election poster on the street is influence.
25:28 
We can critically relate to that: There's a poster, someone wants to sell me a political message,
25:35 
and I can position myself in relation to it. With data-based systems, however, I don't notice
25:41 
what's happening to me, and I can't notice it because they know me better than I know myself.
25:47 
And it goes even further: I'm told stories - and I'd like to ask the expert if they're true:
25:52 
Facial recognition programs can recognize my features, for example analyze depressive moods.
25:58 
This can be coupled with certain word choices that are more common among depressed people. And then the following happens:
26:04 
In my thread, mood-related things are suggested that don't change this mood, but deepen it.
26:11 
Because the companies have realized: Someone who becomes depressed stays longer on these contents, stays longer in the thread.
26:16 
That's an attack on my mental health. - Absolutely. A big problem from a human rights perspective
26:24 
is currently also the concern about the mental health of children, adolescents, and also adults.
26:30 
Because we are exposed to this manipulation almost around the clock
26:35 
and it's no longer possible for us to escape it. We can observe a similar phenomenon
26:42 
unfortunately also with racist hate speech. Technology companies unfortunately fail here
26:47 
to substantially intervene. In the past, this has led to
26:53 
certain people being so incited, through misinformation, so-called fake news...
26:59 
This extends to deepfakes, where I no longer know whether the person ever said that.
27:04 
But I see it as if it were true. They then committed acts of violence in the real world.
27:10 
And we know thanks to whistleblowers from then Facebook now Meta, that the company knew this and had the possibility
27:19 
to intervene and prevent this hate speech online. But Facebook decided
27:26 
not only to let it continue but also to fuel it, because the longer people stay on the platform,
27:33 
the more the company earns. This is a scandal of practical reason,
27:38 
that occurs here daily. There apparently is no lack of factual knowledge about this.
27:44 
And yet - now I use your "we" - we put up with it. There should have long been extensive damage claims
27:54 
that are enforced. Why isn't this happening? Because we have an enforcement problem:
28:00 
We have no institution that takes sufficient care of it. From the complexity, the impression can simultaneously arise
28:08 
that digital transformation, the use of data-based systems is something that has fallen from heaven and is natural
28:16 
and we must endure it. - A fate. Exactly, a fate that comes over us.
28:22 
I would strongly argue that we recognize: We are the ones who can shape this.
28:28 
We can decide how this progress proceeds. Whether we focus on the ethical opportunities of data-based systems
28:36 
and specifically promote them. Or whether we leave it to a few multinational technology companies
28:43 
to serve their individual interests and greatly enrich themselves - at the cost of the rest of humanity and the planet.
28:51 
Would another conjecture regarding the abysmal relationship be that there's also a competency problem?
28:58 
That there's a kind of head start for these technological drivers? The legal medium is not only slow,
29:05 
one also doesn't have the understanding of the inner workings of what's happening there, so that regulation can't be effective.
29:13 
This seems to me too comfortable an excuse for political and economic decision-makers.
29:21 
Because if I steal data, that's a human rights violation. There seems to be no complexity clouding the view there.
29:28 
If I expose young people 7 days, 24 hours a day
29:35 
to a platform like TikTok, which as a Chinese company is very close to the Chinese government,
29:42 
that's certainly not a good idea for the mental health of young people. Plus the data flows elsewhere.
29:49 
Exactly, the data that can then be exploited: That's also something one can quickly understand.
29:55 
The complexity aspect - "it's too technical" - really seems like an excuse to me.
30:01 
In my opinion, there's a lack of political will to appropriately address these issues and get to the point
30:07 
of improving enforcement. We then don't need new regulations,
30:12 
but enforcement of already existing human rights-based standards.
30:18 
Would you say the nations, the governments have been sleeping, and the philosophers too?
30:23 
I wouldn't say that, there is an intensive philosophical discourse on the topic.
30:31 
What we as a scientific discipline should do more
30:37 
is to push for decision-makers - perhaps with the help of research or based on arguments,
30:47 
that we laboriously and carefully develop, and also the insights we gain -
30:53 
to take action. That we try, as with political processes,
30:59 
to better inform about the ethical opportunities and risks of this technology-based innovation.
31:06 
I notice when looking at your profile: You are a Catholic teaching theologian, one can say that.
31:11 
There aren't so many global organizations that also have a certain power in institutions.
31:16 
The Catholic Church would be one of them. Should philosophers and churches join forces
31:22 
to exert institutional pressure? That seems to be your commitment. It certainly wouldn't be wrong.
31:29 
I believe that religious and worldview communities can play an important role,
31:35 
by first creating spaces for people to deal with uncertainty...
31:40 
It doesn't even have to be negatively connoted, but this uncertainty about how it will be in the future -
31:46 
regarding one's own profession, our interpersonal life. Creating spaces where people can exchange about this
31:55 
and also find ways to develop competency to deal with this rapidly changing reality
32:02 
in a meaningful and also purpose-giving way. That seems to me to be one role.
32:07 
But religious and faith communities should also participate in the democratic opinion-forming
32:13 
and decision-making process, to support an ethically based development in the technology sector.
32:21 
We had spoken about a major bias problem - the so-called bias based on datasets.
32:27 
There are other fields of application, namely in the area of social scoring. There, citizens of certain communities or states are recorded,
32:35 
evaluated and cataloged. We associate this primarily with China, and to an almost totalitarian extent.
32:41 
The people there are really clearly categorized based on the data. But similar things exist in Switzerland too.
32:48 
Let's watch a film about this as well. How a typical mass shooter behaves,
32:53 
the algorithm-supported system DyRiAS claims to know. DyRiAS is part of the toolbox
33:01 
of Zurich's Office for Violence Prevention.
33:06 
We never do this alone, always in cooperation with the police, as the police can and may ask very different questions
33:16 
than we can as a specialist office.
33:22 
Together they go through DyRiAS's 35 questions and answer them with "applicable" or "not applicable."
33:30 
Explanation videos help the users with this. A simple algorithm weights the answers
33:37 
and spits out a risk assessment. Which questions are decisive for this judgment
33:43 
is a trade secret of the manufacturer. According to the manufacturer, the instrument was developed through analyses
33:50 
of international cases of violence and mass shootings at schools. In the 20th and 21st centuries
33:57 
several hundred registered mass shootings occurred at schools worldwide, most in the USA.
34:03 
Even if the German developer had analyzed all cases, it is scientifically questionable
34:08 
whether one can create a universally valid profile with which violent acts can be predicted independently of culture.
34:17 
Also an interesting example: The purpose is acknowledged as good, wanting to prevent violence, mass shootings.
34:22 
There is a very narrow data basis, a non-transparent algorithm and apparently a cataloging of persons on this basis.
34:31 
Yes, that's highly dangerous. As you just said: The acts themselves are terrible and one wants to prevent them,
34:38 
but the means seems wrong. For one thing, we don't know if it works. For another, there's the great danger associated with it
34:46 
that people are discriminated against and prejudged wrongly. Unfortunately, this is something where one thinks one could fall for it,
34:54 
because the acts themselves that one wants to prevent are so bad that one absolutely wants to - but the path there is the wrong one.
35:01 
But nevertheless - and this seems to be a fateful movement - we are moving in Western states too
35:08 
toward societies where data collection of persons directly intervenes in social life.
35:14 
For example, my creditworthiness would be the most immediate. But many other things too. I have the feeling
35:21 
that we sometimes talk about China as the great demon, the completely different one who does it, and don't understand
35:28 
that this is already being applied here too. We must criticize the current Chinese government
35:34 
for what they do in terms of total surveillance. At the same time, we must not close our eyes to the fact
35:41 
that the majority of technologies for this don't come from China, but from Europe and the USA.
35:48 
And of course these technologies are also available to governments in Europe and the USA, for example.
35:55 
Therefore, we as citizens should be very vigilant that our governments don't reach into this toolbox.
36:04 
Because it violates the human right to data protection and privacy. Very briefly: Why is it so important
36:10 
to protect data protection and privacy? Because it's relevant to freedom. People behave differently
36:17 
when they know they're being monitored. We then tend to behave in a standardized way:
36:22 
to conform to norms when we know we're being monitored. For example, at the airport we avoid saying certain terms loudly,
36:29 
because we don't want to draw attention. We don't stand there and shout: "Bomb!"
36:35 
when there is no bomb, just for fun. We know that we're being monitored
36:40 
and therefore adapt our behavior. This isn't so tragic in that example, but it shows
36:46 
how we normalize our behavior when we knowingly are being monitored.
36:52 
Now I ask the theologian: That was once the idea of God. He has all the data, looks at it and monitors us -
36:59 
hopefully with the purpose that we behave well. Only the idea back then was already:
37:05 
Is God a loving God, who... That was always a problem with the matter.
37:10 
...who doesn't want to sell us anything or doesn't want to get us to vote for someone specific
37:17 
or make political decisions in their interest. This is also for a Catholic theologian a small mental puzzle,
37:24 
how exactly this relates. In my opinion, the theological-ethical approach
37:30 
can specifically help to recognize more precisely who creates what where and with what abilities or characteristics.
37:41 
Humans have from their freedom the possibility to decide between ethically right, wrong, good or bad
37:49 
and are therefore different from machines in this regard. This is something that is accessible to us in a different way through theological ethics -
37:57 
perhaps not better way - when we also understand humans as created by God in freedom.
38:04 
This can certainly be a contribution that theological ethics can bring to the discourse.
38:10 
Regarding this freedom - also as creative freedom - the air is getting a bit thinner due to AI.
38:17 
For a year now we have had the program ChatGPT, a so-called generative AI that can simulate almost all creative behavior
38:23 
that humans show, based on data. It can write texts, design images, compose music.
38:31 
Humans as creative beings of freedom are under similar pressure. It's also relevant to human rights, if I see correctly,
38:38 
because the data comes from human creators. Exactly, it leads to copyright violations,
38:43 
data protection and privacy violations, as I cannot decide about the use of my data by ChatGPT.
38:49 
It's just done. My informational self-determination is violated. Here too, in my opinion, we need to be careful
38:57 
about what we all read into data-based systems, what they actually aren't.
39:02 
ChatGPT is figuratively speaking actually a ruminating cow. A data parrot. - Exactly.
39:08 
It reproduces what humans have already created, what humans have already thought or written.
39:16 
Often in new combinations, of course, but ultimately going back to human creative work
39:22 
and human creative processes, without acknowledging them. We should be careful
39:28 
and not designate ChatGPT as a new source of knowledge. It does nothing other than what humans have created,
39:35 
ruminating and spitting it out. We need to be careful that we don't interpret more into it
39:41 
than it actually is. I consider generative AI in the image realm
39:46 
in combination with language as very dangerous for democratic systems.
39:53 
Because then one can with so-called deepfakes, that is, with... "Deep deceptions" one could say. - Exactly.
40:00 
There we see people speaking as they normally speak. But they're not really them.
40:06 
Imagine it being spread how a politician says something absurd.
40:12 
And then there comes a tipping point where it has created such an effect
40:18 
that it will then be difficult for the politician to say publicly: "That wasn't me."
40:25 
Because then there's the danger that trust in the political system is lost,
40:30 
trust in this politician. The whole uncertainty: "Was it him or her now or not?"
40:37 
It can go so far that one remains trapped in this false narrative,
40:43 
because the price of getting out of it would be higher than what one subjects oneself to. This goes even further now:
40:50 
It would be possible, based on your computer and the data on it, to create a kind of doubling of you using AI.
40:58 
This would, as a program, write texts that resemble yours. So we're entering the realm
41:03 
where something like an individual human voice can be simulated through these individualized programs so well
41:11 
that almost what makes us human too, uniqueness, becomes foggy. - Exactly.
41:17 
And there we must improve from a legal perspective, because we're still completely blank regarding
41:24 
what this does to our democratic systems in terms of undermining, and ultimately in manipulation.
41:31 
But isn't the demon out of the bottle? We'll never get it back in. I would say that we as humanity have proven
41:39 
that we have often not implemented technically feasible things for ethical reasons.
41:46 
Or only in adapted form within an ethically or legally defined framework.
41:52 
So we have this ability, we just need to quickly unpack it and apply it in the area of data-based systems.
42:00 
Because I agree with you: We don't have much time left for this. If we've now spoken like we have,
42:07 
and we note that there are no rules. No rules, zero, so far.
42:12 
Then one thinks: We are unbelievably stupid. We can't allow this for ten years.
42:20 
I think something like this doesn't just happen by chance. There aren't many,
42:26 
but some quite weighty technology companies that make a lot of money from the fact that we haven't done anything.
42:34 
There's a gold rush atmosphere there. They also know they can't do this forever. So they're trying to get the maximum out now.
42:42 
Fortunately, there is now a growing consensus among UN member states that regulation is needed -
42:49 
in terms of a human rights-based approach and human rights-based data systems. That an International Data-Based Systems Agency,
42:57 
so an international agency for data-based systems IDA... IDA, the name of the authority to be created? - Exactly.
43:05 
Technology companies are also realizing that they shouldn't oppose this.
43:12 
Because they notice that this has already become too strongly established,
43:18 
these two pillars. The next weeks or months will show that attempts are now being made
43:25 
to remove IDA's teeth, so to speak. So you're already sensing: First, your measure has found resonance,
43:31 
the UN Council of Elders has also supported you. A lot is happening right now, and you sense, if I see correctly,
43:38 
that companies are now getting involved and want to water down this wine. Yes. It's encouraging that the UN leadership picked this up and said
43:47 
they wanted this: human rights-based approaches and IDA. Now technology companies realize that it no longer makes sense
43:54 
to resist regulations - that's coming. It makes no sense to resist this institution or IDA.
44:02 
But now they're trying to remove IDA's teeth, that is, to ensure it doesn't become an institution
44:08 
that can really intervene and sanction, that can apply rule-of-law instruments
44:14 
when someone doesn't follow the rules. They want to turn it into a watered-down "expert panel"
44:22 
in the style of the... The example that's often cited as a reference point
44:27 
is the expert council on climate protection. Which of course... That's been working super well in political implementation for 30 years, right?
44:36 
Yes, this panel certainly has its significance and importance, I don't want to criticize it.
44:42 
But we also see there: If an institution has no enforcement possibilities,
44:47 
sanction possibilities or rule-of-law instruments, no pressure means and can't impose fines -
44:54 
as the International Atomic Energy Agency can, then... There are beautiful Sunday sermons from everyone:
45:01 
Everyone has recommendations and guidelines and declarations,
45:06 
there's an abundance of them. Beautiful Sunday sermons, but during the week absolutely nothing happens.
45:14 
We need to get into action here, because it's about human rights violations. And they are so urgent because it's about physical survival
45:22 
of humans and dignified life. So there is urgent need for action.
45:27 
This brings us to the topic of the 75 years we're celebrating: How important it is that there are globally capable institutions
45:34 
that can implement and also sanction these rules. There too - without becoming sad - one could say:
45:41 
The UN has never been in such a poor state regarding rule enforcement.
45:46 
And we are in the process of perceiving how a whole philosophical discourse would say: "Be careful with your human rights."
45:54 
"These are your data-based human rights from Western culture. We don't need them in that way."
46:00 
You're fighting against that too. What we can learn from the Universal Declaration of Human Rights
46:06 
is, for one thing, that we need institutions that can enforce legally established human rights
46:13 
and are equipped with corresponding power. If not, then we have weaknesses in the realization of human rights.
46:20 
Secondly, regarding the global dimension of human rights - the universal validity - it seems important to consider:
46:27 
If we look at the history of human rights - they didn't arise only in a certain region of the world
46:34 
and not in a certain religion, culture, or tradition. Instead, we see the following:
46:41 
In different places of the world, in different cultures, traditions, worldviews, and religions, people were
46:48 
repeatedly confronted with unjust realities and then developed human rights ideas
46:56 
through exchange, discussion, and criticism with the unjust reality. A kind of world ethos, as Hans Küng once called it:
47:02 
That there are traces of human rights discourse in every culture. Exactly, that's interesting.
47:08 
Humans experienced slavery, for example, and realized something was wrong there. "How can we only distinguish between free people and slaves
47:15 
or slaves, we are all equal after all." Thus they developed the demand for equality of humans,
47:21 
which is a human rights idea. This then led to human rights becoming
47:28 
increasingly legally codified from these ideas, because people realized: "If we don't make it positive law,
47:35 
that can also be enforced, then too little changes." "Then people remain victims in their human rights violations
47:41 
and we can't change that." This led to further legalization. Therefore, the thesis that it's something so-called Western is false.
47:50 
It emerged from very different cultures, traditions, religions, and worldviews.
47:55 
Moreover, one can also ask what a geographic category, like West, East, North, South, has to do in a normative discourse.
48:03 
So what epistemological value it has. An argument is usually not good
48:10 
because it was thought of at a particular place at a particular time. These people would perhaps argue analogously:
48:17 
"You have only considered your datasets so far." "Consider our datasets: the cultural experiences."
48:24 
Regarding datasets, I would say yes. Regarding cultural experiences also.
48:29 
There we need human rights as protection to be able to live cultural, religious diversity.
48:34 
It's not something natural, self-evident, that religious, cultural and worldview diversity
48:40 
can be lived and practiced. We need a foundation there too, so that the various religions, cultures,
48:47 
worldviews, philosophies respect each other and don't deny each other's right to exist.
48:53 
That's what human rights contribute to the protection and promotion of diversity.
49:00 
We shouldn't cut off this branch, but nurture it, because it helps us by protecting each and every one of us -
49:08 
in their freedom and self-determination. And only from this can diversity grow.
49:15 
One can see, there are quite different dimensions that interlock or reinforce each other when things go well.
49:22 
There is still a third step, and we are perhaps witnessing it now or will witness it.
49:28 
And that is the question of whether these machines themselves come into systemic states that make us believe
49:36 
that they decide willfully and stubbornly. There are various science fiction scenarios for this,
49:42 
but it can no longer be completely ruled out that we at least come to a point where we can no longer say why and how it made its decision,
49:52 
and whether there was some form of self-reflection. That means, then the machines themselves would be ethical agents,
49:58 
that one could address with ethical commandments. I would say that we should maintain the human-machine separation
50:05 
despite all - even intensive - interaction and merging.
50:11 
Because I would argue that machines lack something to address them as moral or ethical agents:
50:20 
Namely freedom, moral capability, and autonomy, meaning the capacity for self-legislation.
50:27 
Although these concepts need to be clarified first to be able to apply them firmly. Exactly: First of all freedom, as the ability to distinguish between ethically right,
50:36 
wrong, good and bad. Then the idea of moral capability: That I am capable
50:42 
of finding out and recognizing for myself what should and should not be ethically valid.
50:48 
Humans have the ability to set this themselves - in the sense of autonomy, being able to establish laws themselves.
50:55 
I would say: Machines can indeed develop new rules - they can do that now, and can also follow them -
51:02 
but the ethical quality of these rules is not accessible to them. We as humans must be careful that with machines,
51:09 
which can do a lot themselves, we don't read this into them. For the coming years, from an ethical perspective, the crucial thing
51:16 
for the interaction between humans and machines is that we don't read too much into machines with over-trust.
51:22 
Project into them. - Exactly. We must clearly distinguish: What can data-based systems do and what can't they do?
51:30 
What are uniquenesses of humans that we should nurture? Because this will also distinguish us in the future
51:36 
from data-based systems, namely critical thinking, moral capability, creativity -
51:43 
in the sense of authentic creativity. Wouldn't you also say - and I wonder
51:49 
that you haven't said it yet: God's image? That's also an argument that could be brought into play here.
51:56 
One wouldn't say that an AI was created by God in His image, but a religious person believes this.
52:03 
And normatively something follows from that too. Absolutely, that would be something that theological ethics could contribute.
52:10 
Because being made in God's image doesn't just mean that we as humans received this gift,
52:15 
of being created in His image by God. This gift is also connected with a task:
52:21 
To bear responsibility in the sense of this divine image. That means, in God's name to take care of other humans,
52:29 
for the planet. And only humans can bear this responsibility.
52:35 
Let's imagine a self-driving vehicle, for example. If it causes an accident - although to be fair, one must say
52:42 
that self-driving vehicles probably cause fewer accidents than humans...
52:47 
Because with machines, many causes of accidents are eliminated: drunk driving, being distracted, fatigue,
52:54 
being unhappily in love, too much testosterone. These causes don't exist with self-driving vehicles.
52:59 
How imperfect we are! - Yes. Then it makes no sense, if it should cause an accident,
53:07 
to punish it by scrapping it or saying: "Now I won't plug you into power for two weeks."
53:13 
It becomes clear that it makes no sense to transfer responsibility to a machine. The human is needed.
53:19 
We'll know that when the machines say: "I don't want that, you're not doing that to me now."
53:25 
We must keep in mind - and here I agree with you, that in certain areas of intelligence
53:31 
as humans we will fall so massively behind that we must be careful
53:37 
that this superiority in these areas of intelligence - computational ability, logical deduction, handling large amounts of data -
53:45 
doesn't lead to us losing control. We should keep that in mind for the medium or longer term.
53:51 
So you don't think this is silly science fiction talk? It's a potential scenario that could occur?
53:58 
Absolutely, because in the near future we can expect that computational capability in data-based systems will explode.
54:06 
And in areas where it's directly relevant, it naturally leads to corresponding effects.
54:11 
We must keep this in mind because it can happen to us that we get pushed out of certain processes.
54:18 
A current example: There's already a company that offers to serve you a new political party in split seconds
54:25 
with political party program, including all party speeches, all posters, the whole campaign, the social media campaign,
54:32 
in split seconds. Zero cost? - Zero cost, almost no time. And there we must be careful when certain political parties
54:41 
or politicians almost proudly say: "I don't write my speeches anymore, I have them written by a data-based system."
54:49 
That's a testament to the poverty of our politics. When we humans are no longer willing to think
54:54 
and write for ourselves, then we really need to be careful that data-based systems don't take the steering wheel from our hands.
55:04 
You have very correctly and impressively emphasized repeatedly that this is not fate, it's still in our hands.
55:10 
With everything we've seen in the last 20 years - probably only a fraction of what's coming -
55:17 
one could get the idea to simply pull the plug. That we decide not to let this technological branch
55:24 
grow any further. If you as an ethicist get a call from the UN:
55:30 
"We're pulling the plug now." Would you be in favor? I think we should pull the plug in certain areas.
55:39 
There are things we shouldn't do. I'm thinking, for example, of lethal, automated weapon systems.
55:44 
It's foreseeable that it will lead to more wars... - Absolute ban. I would advocate for an absolute ban,
55:51 
because it's foreseeable that it will lead to more war and violence. Because the threshold is lower to start a violent conflict,
55:59 
as one has the feeling one loses nothing except the machines. The costs for this will also sink massively.
56:05 
There I think we should absolutely pull the plug. With other things I would view it more differentially,
56:11 
because we should always keep in mind that thanks to technology-based progress
56:16 
we also achieve ethically positive things. So don't wall up this door. That's always a danger from fear. - Exactly.
56:24 
If the UN were to call me, I would answer: "Let's identify more precisely
56:30 
what are already ethical problems today." "We should eliminate these from the world."
56:36 
"What are future ethical problems and ethical opportunities?" "Let's specifically pursue the opportunities
56:43 
and either avoid or master the problems." Currently we leave it to a handful of technology companies.
56:52 
Such a concentration of political and economic power has never been experienced in human history.
56:58 
We leave it to them how things go. And that's really the wrong place.
57:03 
Because the technology companies are primarily concerned with increasing their own profit, not with ensuring
57:11 
that humanity and the planet have a future. Mr. Kirchschläger, I believe we share the feeling
57:16 
that the discussion about this has only begun, is important and will accompany us for a long time. If we do it with natural intelligence - like you -,
57:24 
it will be all the more valuable. Thank you for the conversation. - Thank you.
57:32 
What ethical rules should artificial intelligence absolutely be subject to? Suggestions welcome in the comments section. You can find another conversation about AI and the chatbot revolution here."""

    srt_output = convert_to_srt(sample_text)
    print(srt_output)