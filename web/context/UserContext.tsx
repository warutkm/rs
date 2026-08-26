'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { DemoUser, DEMO_USERS, DEFAULT_USER } from '@/lib/demoUsers';
import { RecSysAPI } from '@/lib/api';

interface UserContextType {
  currentUser: DemoUser;
  setCurrentUser: (user: DemoUser) => void;
  setUserById: (userId: string) => void;
  allUsers: DemoUser[];
  cartCount: number;
  addToCart: (itemId: string, itemTitle?: string) => void;
  likedItems: Set<string>;
  toggleLike: (itemId: string) => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

export function UserProvider({ children }: { children: React.ReactNode }) {
  const [currentUser, setCurrentUser] = useState<DemoUser>(DEFAULT_USER);
  const [cartCount, setCartCount] = useState<number>(0);
  const [likedItems, setLikedItems] = useState<Set<string>>(new Set());

  // Restore user from localStorage if available
  useEffect(() => {
    try {
      const savedUserId = localStorage.getItem('recsys_demo_user_id');
      if (savedUserId) {
        const found = DEMO_USERS.find((u) => u.id === savedUserId);
        if (found) {
          setCurrentUser(found);
        } else {
          setCurrentUser({
            id: savedUserId,
            name: `Custom User (${savedUserId.slice(0, 8)}...)`,
            persona: 'Custom Seeded Profile',
            avatar: '👤',
            badgeColor: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/30',
            description: 'Custom user identifier passed directly to candidate retrieval models.',
            seedPreferences: ['Custom Preferences']
          });
        }
      }
    } catch {}
  }, []);

  const handleSetCurrentUser = (user: DemoUser) => {
    setCurrentUser(user);
    try {
      localStorage.setItem('recsys_demo_user_id', user.id);
    } catch {}
  };

  const setUserById = (userId: string) => {
    const found = DEMO_USERS.find((u) => u.id === userId);
    if (found) {
      handleSetCurrentUser(found);
    } else {
      handleSetCurrentUser({
        id: userId,
        name: `Custom User (${userId.slice(0, 8)}...)`,
        persona: 'Custom User ID',
        avatar: '👤',
        badgeColor: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
        description: 'Direct custom user ID for ranker candidate scoring.',
        seedPreferences: ['Direct Mode']
      });
    }
  };

  const addToCart = (itemId: string, itemTitle?: string) => {
    setCartCount((prev) => prev + 1);
    RecSysAPI.logEvent({
      user_id: currentUser.id,
      item_id: itemId,
      event_type: 'cart',
      metadata: { item_title: itemTitle, timestamp: new Date().toISOString() },
    });
  };

  const toggleLike = (itemId: string) => {
    setLikedItems((prev) => {
      const next = new Set(prev);
      const isLiked = next.has(itemId);
      if (isLiked) {
        next.delete(itemId);
      } else {
        next.add(itemId);
      }
      RecSysAPI.logEvent({
        user_id: currentUser.id,
        item_id: itemId,
        event_type: 'rating',
        rating: isLiked ? 3.0 : 5.0,
        metadata: { liked: !isLiked },
      });
      return next;
    });
  };

  return (
    <UserContext.Provider
      value={{
        currentUser,
        setCurrentUser: handleSetCurrentUser,
        setUserById,
        allUsers: DEMO_USERS,
        cartCount,
        addToCart,
        likedItems,
        toggleLike,
      }}
    >
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}
