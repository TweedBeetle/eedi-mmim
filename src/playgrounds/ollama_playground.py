import ollama

ollama.embed(model='nomic-embed-text', input='The sky is blue because of rayleigh scattering')

print(ollama.generate(model='phi3.5', prompt='Why is the sky blue?'))
