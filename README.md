# BiasNET

This is a group polarization simulator that operates off of a few fundamental principles. 

This is the basic algorithm: 
1. Create agents
2. Create a list of arbitrary issues, numbered issue1, issue2, ... ,issue n. 
3. Let agents have "belief" for or against issues, on a scale of -1 for "against" to 1 for "for", with 0 being neutral. Initialize this belief randomly.
4. Make agents have "affinity" towards each other, which starts off at 0
5. Make agents with similar beliefs on the same issues increase affinity for each other with each chronological step, and agents with dissimilar beliefs decrease affinity with each other with chronological step
6. With each step modify beliefs and affinity based on each other
7. Observe if a number of steps into this simulation, we have polarized groups - or, in mathematical terms, if there's a strong correlation between beliefs in one issue and another. Do most agents that are *for* issue 1 *against* issue 2? Or something along those lines. If that holds true, then we can say that small fluctuations plus reactions do force polarization. This doesn't say anything about human behaviour, because we haven't yet linked this model's fundamental axioms with those of human behaviour, but it's an interesting observation nonetheless.

You can play around with this [here](https://biasnet.streamlit.app).