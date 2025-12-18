  # ┌──────────────────────────────┐
  # │         GROUND TRUTHS        │
  # └──────────────────────────────┘

video_1_ground_truth = """
====Important Details====
- Nighttime police activity
- Officers escort a handcuffed suspect into a white van
- Suspect is secured inside the van with a seatbelt
====Auxillary Details====
- Suspect wears a blue-and-yellow beanie, gray polkadot shirt, and black pants
- An officer uses a flashlight
- A female officer finishes fastening the seatbelt
- The suspect seems to be cooperative with the officers
"""

video_2_ground_truth = """
====Important Details====
- Officers with weapons drawn are shown to approach a person
- Shooter fires first, Officer mentions "Shots Fired"
- Officers attempt subdue the shooter on top of a parked car
- Recording officer tells the shooter to "Drop the fucking gun"
====Auxillary Details====
- Officer communicates over police radio with dispatch
- Scene occurs on a dark city street with parked cars and dumpsters
- Taser sound can be heard
"""

video_3_ground_truth = """
====Important Details====
- Starts with a traffic stop
- Officer asks to see hands and the suspect reaches for something
- Officers back off and say "Let me see your hands", Officer weapons are drawn
- Officer fires, gun visible in frame, says "Shots fired", "Shots fired by police"
====Auxillary Details====
- Multiple officers are present besides the POV officer
- Officer’s firearm is visible in frame
"""

video_4_ground_truth = """
====Important Details====
- Officer speaks with a suspect in a vehicle
- Suspect suddenly flees
- Officer pursues in a police cruiser
- Suspect crashes into a pole
- The vehicle is heavily damaged and smoking
====Auxillary Details====
- Officer reports the incident over the radio
- Vehicle is a blue SUV
"""

video_5_ground_truth = """
====Important Details====
- Officers respond to a civilian argument that escalates into violence
- A shooting occurs during the incident
- Officers identify an armed suspect
- Suspect is shot and taken down
- Officers communicate details to dispatch
====Auxillary Details====
- Location is reported as Waveland and Marshfield
- Civilians argue prior to the shooting
- A bystander reports the suspect’s description
- A jacket is shown lying on the ground
"""

video_6_ground_truth = """
====Important Details====
- Officers approach two armed individuals with raised hands
- One individual shows a holstered firearm
- He identifies himself as a police officer
- He states he fired shots during an altercation
- Officers request medical assistance
====Auxillary Details====
- Clothing descriptions of the individuals
- One individual points out where shots were heard
- Dispatch asks whether the individuals are compliant
"""

video_7_ground_truth = """
====Important Details====
- Officer conducts a traffic stop for vehicle violations
- Suspect disputes the reason for the stop
- Officers order the suspect out of the vehicle
- Suspect is taken to the ground and handcuffed
====Auxillary Details====
- Second officer assists during the stop
- Suspect repeatedly says “Are you serious right now”
- Suspect asks for his mother
"""

video_8_ground_truth = """
====Important Details====
- Officers place an unresponsive or limp man into a police vehicle
- Multiple officers secure the scene
- Officers manage nearby bystanders
- Officers state an ambulance will be called
====Auxillary Details====
- Bystanders verbally react to the arrest
- A female officer orders people to leave the area
- A bystander claims the man was shot
"""

video_9_ground_truth = """
====Important Details====
- Officer aims a firearm at a vehicle believed to contain an assailant
- Officer fires multiple shots at the vehicle
- Assailant flees on foot
- Officers pursue and arrest the assailant
- An officer is injured and EMS is requested
====Auxillary Details====
- Officer initially points weapon toward a nearby house
- Officer damages a civilian vehicle during pursuit
- Officers verbally accuse the suspect of shooting at them
"""

video_10_ground_truth = """
====Important Details====
- Nighttime police response on an urban street
- Officers order people to leave the area
- One individual refuses to comply
- Officers take the individual to the ground and arrest them
====Auxillary Details====
- Officers use flashlights
- Several bystanders are present
- Arrestee claims to know people inside a house
"""

video_11_ground_truth = """
====Important Details====
- Nighttime police interaction outside a residence
- Officer speaks with an intoxicated individual
- Another person claims the individual should be arrested
- Keys are located, resolving the immediate issue
====Auxillary Details====
- A translator assists one individual
- The individual states he does not want trouble
- A bystander mentions the keys were in another vehicle
"""


# Video ground truth dict
copa_video_ground_truths = {
   "video1": video_1_ground_truth,
   "video2": video_2_ground_truth,
   "video3": video_3_ground_truth,
   "video4": video_4_ground_truth,
   "video5": video_5_ground_truth,
   "video6": video_6_ground_truth,
   "video7": video_7_ground_truth,
   "video8": video_8_ground_truth,
   "video9": video_9_ground_truth,
   "video10": video_10_ground_truth,
   "video11": video_11_ground_truth
}
