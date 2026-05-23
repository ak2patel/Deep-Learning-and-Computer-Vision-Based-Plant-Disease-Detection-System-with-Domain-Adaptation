"""
Authentication routes: signup, login, and user profile.

Endpoints:
    POST /api/auth/signup  - Create a new user account
    POST /api/auth/login   - Login and receive JWT token
    GET  /api/auth/me      - Get current user profile (requires JWT)
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from utils.database import db
from utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

router = APIRouter()


# ---------- Request / Response Schemas ----------

class SignupRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    created_at: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ---------- Endpoints ----------

@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def signup(request: SignupRequest):
    """
    Create a new user account.
    
    - Validates that the email is not already taken
    - Hashes the password with bcrypt
    - Returns a JWT access token
    """
    # Check if user already exists
    existing_user = await db.user.find_unique(where={"email": request.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email already exists"
        )
    
    # Validate password length
    if len(request.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters long"
        )
    
    # Create user
    hashed = hash_password(request.password)
    user = await db.user.create(
        data={
            "email": request.email,
            "name": request.name,
            "passwordHash": hashed,
        }
    )
    
    # Generate JWT
    token = create_access_token(data={"sub": user.id})
    
    return AuthResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.createdAt.isoformat(),
        )
    )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Login with email and password.
    
    - Verifies credentials against the database
    - Returns a JWT access token on success
    """
    # Find user by email
    user = await db.user.find_unique(where={"email": request.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(request.password, user.passwordHash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate JWT
    token = create_access_token(data={"sub": user.id})
    
    return AuthResponse(
        access_token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            created_at=user.createdAt.isoformat(),
        )
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user=Depends(get_current_user)):
    """
    Get the profile of the currently authenticated user.
    
    Requires: Authorization: Bearer <token>
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.createdAt.isoformat(),
    )
