'use client';

import React, { useState, useRef, useEffect } from 'react';
import { useUser } from '@/context/UserContext';
import { DemoUser } from '@/lib/demoUsers';
import { Users, ChevronDown, Check, Sparkles, UserCheck, ShieldAlert } from 'lucide-react';

export default function UserSwitcher() {
  const { currentUser, setCurrentUser, setUserById, allUsers } = useUser();
  const [isOpen, setIsOpen] = useState(false);
  const [customInput, setCustomInput] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (user: DemoUser) => {
    setCurrentUser(user);
    setIsOpen(false);
  };

  const handleCustomSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (customInput.trim()) {
      setUserById(customInput.trim());
      setCustomInput('');
      setIsOpen(false);
    }
  };

  const isColdStart = currentUser.id === 'guest_cold_start';

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-700/80 hover:border-sky-500/50 hover:bg-slate-800 transition-all text-left shadow-sm group"
        title="Switch active user profile to see personalization change"
      >
        <span className="text-base select-none">{currentUser.avatar}</span>
        <div className="flex flex-col">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold text-white group-hover:text-sky-300 transition-colors">
              {currentUser.name.split(' ')[0]}
            </span>
            <span
              className={`text-[9px] font-medium px-1.5 py-0.2 rounded border ${
                isColdStart
                  ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                  : 'bg-sky-500/10 text-sky-300 border-sky-500/30'
              }`}
            >
              {isColdStart ? 'Cold Start' : 'Personalized'}
            </span>
          </div>
          <span className="text-[10px] text-slate-400 truncate max-w-[110px] sm:max-w-[150px]">
            {currentUser.persona}
          </span>
        </div>
        <ChevronDown
          className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${
            isOpen ? 'rotate-180 text-sky-400' : ''
          }`}
        />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-80 sm:w-96 rounded-xl bg-slate-900 border border-slate-700 shadow-2xl z-50 overflow-hidden animate-slide-up">
          <div className="p-3 bg-gradient-to-r from-slate-900 via-indigo-950/40 to-slate-900 border-b border-slate-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
                <Users className="w-3.5 h-3.5 text-sky-400" />
                <span>Demo User Persona Switcher</span>
              </div>
              <span className="text-[10px] text-slate-400">
                Live Ranker Signals
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-1">
              Select a persona to test how LambdaMART ranking & Two-Tower retrieval adapt in real time.
            </p>
          </div>

          {/* User List */}
          <div className="max-h-72 overflow-y-auto p-2 space-y-1 divide-y divide-slate-800/40">
            {allUsers.map((user) => {
              const isSelected = user.id === currentUser.id;
              return (
                <button
                  key={user.id}
                  onClick={() => handleSelect(user)}
                  className={`w-full flex items-start gap-3 p-2 rounded-lg text-left transition-all ${
                    isSelected
                      ? 'bg-sky-950/50 border border-sky-500/40 shadow-inner'
                      : 'hover:bg-slate-800/70 border border-transparent'
                  }`}
                >
                  <span className="text-2xl pt-0.5 select-none">{user.avatar}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-100 truncate">
                        {user.name}
                      </span>
                      {isSelected && (
                        <Check className="w-3.5 h-3.5 text-sky-400 shrink-0 ml-1" />
                      )}
                    </div>
                    <span className="text-[11px] font-medium text-sky-300 block truncate">
                      {user.persona}
                    </span>
                    <p className="text-[10px] text-slate-400 line-clamp-2 mt-0.5 leading-relaxed">
                      {user.description}
                    </p>
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {user.seedPreferences.map((pref, i) => (
                        <span
                          key={i}
                          className="text-[9px] px-1.5 py-0.2 rounded bg-slate-800 text-slate-300 border border-slate-700"
                        >
                          {pref}
                        </span>
                      ))}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Custom User ID Form */}
          <div className="p-3 bg-slate-950 border-t border-slate-800">
            <form onSubmit={handleCustomSubmit} className="space-y-1.5">
              <label className="text-[10px] uppercase tracking-wider font-semibold text-slate-400 block">
                Or Enter Custom `user_id`
              </label>
              <div className="flex gap-1.5">
                <input
                  type="text"
                  value={customInput}
                  onChange={(e) => setCustomInput(e.target.value)}
                  placeholder="e.g. AE3RQLFSVY5DO..."
                  className="flex-1 px-2.5 py-1 text-xs bg-slate-900 border border-slate-700 rounded-md text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500"
                />
                <button
                  type="submit"
                  className="px-3 py-1 text-xs font-semibold bg-sky-600 hover:bg-sky-500 text-white rounded-md transition-colors"
                >
                  Set
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
