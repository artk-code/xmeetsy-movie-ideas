import cohere
import streamlit as st
import os
import textwrap

# Cohere API key
api_key = os.environ["CO_KEY"]

# Set up Cohere client
co = cohere.Client(api_key)

response = co.generate(
  model='command-xlarge-nightly',
  prompt='Generate a movie idea based on the title of two movies seperated by the words meets.\n###\nJaws Meets Twister\n###\nA group of scientists are investigating a mysterious phenomenon that is causing massive destruction in the ocean. As they investigate, they discover that it is a giant, man-eating shark that is responsible for the destruction. They must race against time to stop the shark before it destroys the entire ocean via the world\'s greatest megastorm a tornado on the water!\n###\nSanta Claus Meets Nightmare on ElmStreet\n###\nThe story of Santa Claus, the legendary figure who brings joy to children around the world, meets the horror of the Nightmare on Elm Street. Santa Claus is a kind and gentle man who is always looking out for the best interests of children. He is a protector and a guardian, and he will do anything to keep the children of the world safe. However, when Santa Claus comes face to face with the nightmare of the Elm Street, he must confront his own fears and fight for the children he has vowed to protect.\n###\nTop Gun Meets Transformers\n###\nThe story of Top Gun, the iconic 80s movie about a group of young pilots, meets the world of the Transformers. The Top Gun program has been reactivated and a new group of young pilots are being trained to take on the latest threat to America\'s national security: the Decepticons. The Top Gun pilots must learn to work together and use their skills to take down the Decepticons and protect America from destruction.\n###\nSuper Mario Brothers meets Ant Man\n###\nThe story of the Super Mario Brothers, the classic video game about two plumbers who must save the Mushroom Kingdom from the evil Bowser, meets the world of Ant Man. The Super Mario Brothers are back and they must once again save the Mushroom Kingdom from the evil Bowser. However, this time they are joined by Ant Man, the tiny hero who can shrink down to the size of an ant and use his super strength to take on even the biggest of enemies. Together, the Super Mario Brothers and Ant Man must work together to defeat Bowser and save the Mushroom Kingdom once and for all.\n###\n',
  max_tokens=300,
  temperature=0.9,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))