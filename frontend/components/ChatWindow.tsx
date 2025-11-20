import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

interface Message {
    sender: 'user' | 'agent';
    text: string;
}

export default function ChatWindow() {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Fetch welcome message on open
    useEffect(() => {
        if (isOpen && messages.length === 0) {
            setLoading(true);
            axios.get('/api/agent/welcome')
                .then(res => {
                    setMessages([{ sender: 'agent', text: res.data.message }]);
                })
                .catch(() => {
                    setMessages([{ sender: 'agent', text: "Hi! I'm your Deal Hunter. How can I help?" }]);
                })
                .finally(() => setLoading(false));
        }
    }, [isOpen]);

    const sendMessage = async () => {
        if (!input.trim()) return;

        const userMsg = input;
        setMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
        setInput('');
        setLoading(true);

        try {
            const res = await axios.post('/api/agent/chat', { query: userMsg });
            setMessages(prev => [...prev, { sender: 'agent', text: res.data.response }]);
        } catch (err) {
            setMessages(prev => [...prev, { sender: 'agent', text: "Sorry, I'm having trouble connecting right now." }]);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="fixed bottom-6 right-6 z-50 font-sans">
            {/* Chat Window */}
            {isOpen && (
                <div className="bg-white w-96 h-[500px] rounded-2xl shadow-2xl flex flex-col mb-4 border border-gray-200 overflow-hidden animate-fade-in-up">
                    {/* Header */}
                    <div className="bg-indigo-600 p-4 text-white flex justify-between items-center">
                        <div className="flex items-center gap-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                            <h3 className="font-bold">Deal Hunter AI</h3>
                        </div>
                        <button onClick={() => setIsOpen(false)} className="text-white hover:text-gray-200">✕</button>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`mb-4 flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                                <div className={`max-w-[80%] p-3 rounded-2xl text-sm whitespace-pre-wrap ${msg.sender === 'user'
                                        ? 'bg-indigo-600 text-white rounded-br-none'
                                        : 'bg-white text-gray-800 shadow-sm border border-gray-100 rounded-bl-none'
                                    }`}>
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        {loading && (
                            <div className="flex justify-start mb-4">
                                <div className="bg-gray-200 p-3 rounded-2xl rounded-bl-none animate-pulse text-xs text-gray-500">
                                    Thinking...
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <div className="p-4 bg-white border-t border-gray-100">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                                placeholder="Ask about decks, investments..."
                                className="flex-1 border border-gray-300 rounded-full px-4 py-2 text-sm focus:outline-none focus:border-indigo-500"
                            />
                            <button
                                onClick={sendMessage}
                                className="bg-indigo-600 text-white rounded-full w-10 h-10 flex items-center justify-center hover:bg-indigo-700 transition-colors"
                            >
                                ➤
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Toggle Button */}
            {!isOpen && (
                <button
                    onClick={() => setIsOpen(true)}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-transform hover:scale-110"
                >
                    <span className="text-2xl">🤖</span>
                </button>
            )}
        </div>
    );
}
