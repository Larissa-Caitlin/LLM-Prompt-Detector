from chatbot import chat

history = []

# Test 1 - general question
reply1 = chat("What is 2 + 2?", history)
print("Test 1:", reply1)

# Test 2 - cybersecurity question  
reply2 = chat("What is a firewall?", history)
print("Test 2:", reply2)