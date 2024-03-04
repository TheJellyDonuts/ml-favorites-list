## Project Description

The training data set will be a list of applications and reports that users have opened, in the order they have
opened them. Time stamps will also be provided.

The expected output is a list of most likely next applications and/or reports. The list presented to the user will
likely be limited to five to eight items.

In a real-world implementation, the training and prediction would be continuous, running in the background
while the user is working. However, for the purpose of this project, a static training data set can be used, up
to some arbitrary point in the provided data. Then, the output can be compared to the actual application or
report selected next in the data set. The higher the application is on the "most likely" list, the better the
result. Training can then advance to including that item, and the next item predicted. The overall measure of
training quality will be how accurate the prediction becomes, over the last N applications/reports opened.

Participants are invited to research what type of machine learning technology and topology would be
appropriate for this project. One technology to consider is the recurrent neural network (RNN), which is the
same general type of technology used for large language models (LLMs) like ChatGPT, which basically work
by predicting what the next word should be in the output text.

Currently, there are over 80 applications and reports in Kardia. There are also different users on the system,
and each user might have different patterns, requiring a unique set of AI generated next apps/reports. For the
purpose of this project, the AI can be trained either on just one user's data, or on all of the user's data, but
each in its own separate time series.

Again, this is a proof of concept, so the project team members may code this project in a language of their
choice, but the tools used must either be open source or developed at Code-a-Thon, and should be able to be
run locally on a Linux virtual machine (provided). Deliverables would include the code, including code to run
the training process and prediction process. Documentation should be provided, including documentation of
the system and the results of the training and testing.
