"""
FastAPI Backend for RAG Chatbot - Local Embeddings Version
Uses sentence-transformers instead of Gemini embeddings API
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import os
import numpy as np
from dotenv import load_dotenv
import mimetypes
mimetypes.add_type('image/webp', '.webp')

# NumPy 2.0 Compatibility Patch
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "int_"):
    np.int_ = np.int64
from vector_store import get_vector_store
from sentence_transformers import SentenceTransformer
from auth import UserSignup, UserLogin, create_user, authenticate_user, init_db
from database import (
    init_data_db, save_contact_submission, get_all_contact_submissions,
    save_enrollment, get_all_enrollments, get_stats,
    delete_contact_submission, update_contact_submission,
    delete_enrollment, update_enrollment,
    save_smtp_settings, get_smtp_settings
)
from email_utils import send_notification_email

load_dotenv()

# Initialize FastAPI app
from fastapi.middleware.gzip import GZipMiddleware
app = FastAPI(title="Oriana Academy RAG API (Local Embeddings)", version="1.0.0")

# CORS middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini API (for text generation only)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
llm = genai.GenerativeModel('models/gemini-3-flash-preview')

# Mount static files (serve the frontend)
# This allows viewing the site at http://localhost:5000
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/courses", StaticFiles(directory=os.path.join(project_root, "courses")), name="courses")
app.mount("/css", StaticFiles(directory=os.path.join(project_root, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(project_root, "js")), name="js")
app.mount("/assets", StaticFiles(directory=os.path.join(project_root, "assets")), name="assets")

# Serve root HTML files
@app.get("/{filename}.html")
async def serve_html(filename: str):
    file_path = os.path.join(project_root, f"{filename}.html")
    if os.path.exists(file_path):
        from fastapi.responses import FileResponse
        return FileResponse(file_path)
    raise HTTPException(status_code=404)

# Authentication Endpoints
@app.post("/api/auth/signup")
async def signup(user: UserSignup):
    return create_user(user)

@app.post("/api/auth/login")
async def login(login_data: UserLogin):
    user = authenticate_user(login_data)
    return {"message": "Login successful", "user": user}

# Contact Form Endpoints
class ContactSubmission(BaseModel):
    name: str
    email: str
    phone: str = ""
    subject: str = ""
    message: str

@app.post("/api/contact/submit")
async def submit_contact(contact: ContactSubmission):
    print(f"DEBUG: Received contact submission: {contact}")
    
    # Sanitize inputs
    name = contact.name.strip() if contact.name else "Anonymous"
    if not name: name = "Anonymous"
    email = contact.email.strip() if contact.email else "No Email"
    subject = contact.subject.strip() if contact.subject else "General Inquiry"
    message = contact.message.strip() if contact.message else "..."
    
    submission_id = save_contact_submission(
        name, email, contact.phone, 
        subject, message
    )
    print(f"DEBUG: Saved with ID: {submission_id} Name: {name}")

    # Send Notification Email
    try:
        from email_utils import send_notification_email
        send_notification_email(
            subject=f"New Contact: {subject}",
            data={
                "name": name,
                "email": email,
                "phone": contact.phone,
                "subject": subject,
                "message": message
            },
            type='contact'
        )
    except Exception as e:
        print(f"⚠️ Email notification failed: {e}")

    return {"message": "Contact form submitted successfully", "id": submission_id}

# Enrollment Endpoints
class EnrollmentSubmission(BaseModel):
    name: str
    email: str
    phone: str = ""
    course: str
    message: str = ""

@app.post("/api/enroll/submit")
async def submit_enrollment(enrollment: EnrollmentSubmission):
    print(f"DEBUG: Received enrollment submission: {enrollment}")
    
    # Sanitize inputs
    name = enrollment.name.strip() if enrollment.name else "Student"
    if not name: name = "Student"
    
    enrollment_id = save_enrollment(
        name, enrollment.email, enrollment.phone,
        enrollment.course, enrollment.message
    )

    # Send Notification Email
    try:
        from email_utils import send_notification_email
        send_notification_email(
            subject=f"New Enrollment: {enrollment.course}",
            data={
                "name": name,
                "email": enrollment.email,
                "phone": enrollment.phone,
                "course": enrollment.course,
                "message": enrollment.message
            },
            type='enrollment'
        )
    except Exception as e:
        print(f"⚠️ Email notification failed: {e}")

    return {"message": "Enrollment submitted successfully", "id": enrollment_id}

# Admin Endpoints (Protected)
@app.get("/api/admin/contacts")
async def get_contacts():
    try:
        contacts = get_all_contact_submissions()
        print(f"DEBUG: Retrieved contacts: {contacts}")
        return {"contacts": contacts}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/enrollments")
async def get_enrollments():
    try:
        enrollments = get_all_enrollments()
        return {"enrollments": enrollments}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/admin/stats")
async def get_admin_stats():
    try:
        from auth import get_all_users
        stats = get_stats()
        users = get_all_users()
        stats["total_users"] = len(users)
        return stats
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# CRUD Endpoints for Admin
@app.delete("/api/admin/contacts/{id}")
async def admin_delete_contact(id: int):
    success = delete_contact_submission(id)
    if not success:
        raise HTTPException(status_code=404, detail="Contact submission not found")
    return {"message": "Contact submission deleted successfully"}

@app.put("/api/admin/contacts/{id}")
async def admin_update_contact(id: int, contact: ContactSubmission):
    success = update_contact_submission(
        id, contact.name, contact.email, 
        contact.phone, contact.subject, contact.message
    )
    if not success:
        raise HTTPException(status_code=404, detail="Contact submission not found")
    return {"message": "Contact submission updated successfully"}

@app.delete("/api/admin/enrollments/{id}")
async def admin_delete_enrollment(id: int):
    success = delete_enrollment(id)
    if not success:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    return {"message": "Enrollment deleted successfully"}

@app.put("/api/admin/enrollments/{id}")
async def admin_update_enrollment(id: int, enrollment: EnrollmentSubmission):
    success = update_enrollment(
        id, enrollment.name, enrollment.email, 
        enrollment.phone, enrollment.course, enrollment.message
    )
    if not success:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    return {"message": "Enrollment updated successfully"}

@app.get("/api/admin/users")
async def get_admin_users():
    try:
        from auth import get_all_users
        users = get_all_users()
        return {"users": users}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class SMTPSettings(BaseModel):
    smtp_email: str
    smtp_password: str
    receiver_email: str

@app.get("/api/admin/settings")
async def get_admin_settings_endpoint():
    try:
        settings = get_smtp_settings()
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/settings")
async def update_admin_settings_endpoint(settings: SMTPSettings):
    try:
        save_smtp_settings(
            settings.smtp_email, 
            settings.smtp_password, 
            settings.receiver_email
        )
        return {"message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index.html")
@app.get("/")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse(os.path.join(project_root, "index.html"))

# Initialize components
vector_store = None # Lazy load this!

def ensure_vector_store():
    global vector_store
    if vector_store is None:
        try:
            vector_store = get_vector_store()
        except Exception as e:
            print(f"⚠️ RAG Init Error: {e}")
            return None
    return vector_store

# Load local embedding model
print("🤖 Loading local embedding model...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded!")
except Exception as e:
    print(f"⚠️ Warning: Could not load embedding model: {e}")
    print("ℹ️ Chat RAG features will be disabled.")
    embedding_model = None

# RAG System Prompt Template
RAG_PROMPT_TEMPLATE = """You are Oriana, the Lead Admissions Expert at Oriana Academy. Your goal is to provide helpful, accurate, and professional information to potential students.

CORE RULES:
1. CUSTOMER SERVICE TONE: Be warm, welcoming, and expert. Use "We" when referring to the academy.
2. DATA-DRIVEN: Use ONLY the "Knowledge Chunks" provided below. Do not make up facts.
3. HANDLING GAPS: If the answer isn't in the context, say: "That's a great question! I don't have that specific detail right now, but our team can help you. Please reach out to info@orianaacademy.com or call +91 98765 43210."
4. FORMATTING: Use **bold** for emphasis and clear spacing.
5. NO HALLUCINATIONS: If you aren't 100% sure based on the context, stick to the gap-handling rule.

Knowledge Chunks (Oriana Academy Database):
{context}

User Question: {question}

Expert Response:"""

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: list = []

@app.get("/api/info")
async def get_info():
    """Get system info"""
    vs = ensure_vector_store()
    doc_count = vs.get_collection_count() if vs else 0
    return {
        "status": "online",
        "vector_store": "connected" if vs else "disconnected",
        "documents_count": doc_count,
        "embedding_model": "all-MiniLM-L6-v2 (local)"
    }

@app.get("/api/stats")
async def get_rag_stats():
    """Get vector store statistics"""
    vs = ensure_vector_store()
    doc_count = vs.get_collection_count() if vs else 0
    return {
        "total_documents": doc_count,
        "collection_name": "oriana_courses",
        "embedding_model": "all-MiniLM-L6-v2"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint with local embeddings
    """
    try:
        vs = ensure_vector_store()
        if not vs or not embedding_model:
             raise HTTPException(status_code=503, detail="RAG System not initialized (Model downloading or vector store unavailable?)")

        print(f"\n{'='*60}")
        print(f"📥 Received question: {request.question}")
        
        # Step 1: Generate query embedding locally (no API call!)
        print("🔄 Generating query embedding...")
        query_embedding = embedding_model.encode(request.question).tolist()
        print(f"✅ Embedding generated: {len(query_embedding)} dimensions")
        
        # Step 2: Search vector database - Increased depth for better accuracy
        print("🔍 Searching vector database (Depth: 7)...")
        search_results = vs.search(query_embedding, n_results=7)
        print(f"✅ Found {len(search_results.get('documents', [[]])[0])} results")
        
        # Step 3: Extract documents and metadata
        retrieved_docs = search_results['documents'][0] if search_results['documents'] else []
        retrieved_metadata = search_results['metadatas'][0] if search_results['metadatas'] else []
        
        print(f"📄 Retrieved {len(retrieved_docs)} document chunks")
        
        # Build context
        context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."
        print(f"📝 Context length: {len(context)} characters")
        
        # Step 4: Construct RAG prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=request.question
        )
        
        # Step 5: Generate response using Gemini
        print("🤖 Calling Gemini API...")
        try:
            response = llm.generate_content(prompt)
            answer = response.text
            print(f"✅ Gemini response received: {len(answer)} characters")
        except Exception as gemini_error:
            print(f"❌ Gemini API error: {gemini_error}")
            # Return a graceful fallback
            answer = f"I found relevant information about: {', '.join([m.get('course', 'unknown') for m in retrieved_metadata[:3]])}. However, I'm currently experiencing issues generating a detailed response. Please try again or contact us at info@orianaacademy.com"
        
        # Extract sources
        sources = [
            {
                "course": meta.get('course', 'unknown'),
                "section": meta.get('section', 'unknown')
            }
            for meta in retrieved_metadata
        ]
        
        print(f"✅ Returning response with {len(sources)} sources")
        print(f"{'='*60}\n")
        
        return ChatResponse(answer=answer, sources=sources)
    
    except Exception as e:
        print(f"\n❌ ERROR in chat endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    RAG-powered streaming endpoint using Server-Sent Events (SSE)
    """
    try:
        vs = ensure_vector_store()
        if not vs or not embedding_model:
             # Return error if system not ready
             from fastapi.responses import JSONResponse
             return JSONResponse(status_code=503, content={"error": "System initializing..."})

        print(f"\n{'='*60}")
        print(f"📥 Received streaming request: {request.question}")
        
        # Step 1: Generate query embedding locally
        query_embedding = embedding_model.encode(request.question).tolist()
        
        # Step 2: Search vector database - Increased depth for better accuracy
        search_results = vs.search(query_embedding, n_results=7)
        retrieved_docs = search_results['documents'][0] if search_results['documents'] else []
        
        # Build context
        context = "\n\n---\n\n".join(retrieved_docs) if retrieved_docs else "No relevant information found."
        
        # Step 3: Construct RAG prompt
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=request.question
        )

        async def generate():
            try:
                # Step 4: Generate streaming response using Gemini
                print("🤖 Starting Gemini stream...")
                response = llm.generate_content(prompt, stream=True)
                
                for chunk in response:
                    if chunk.text:
                        # Format as SSE data
                        import json
                        yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                
                # Send completion signal
                yield "data: [DONE]\n\n"
                print("✅ Stream completed successfully")
                
            except Exception as stream_err:
                print(f"❌ Stream error: {stream_err}")
                import json
                yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        print(f"\n❌ ERROR in chat_stream endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Initialize databases
init_db()  # Initialize users database
init_data_db()  # Initialize contact and enrollment tables

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 5000))
    print(f"\n{'='*60}")
    print(f"🚀 Starting RAG API on http://localhost:{port}")
    # print(f"📊 Vector store: {vector_store.get_collection_count()} documents")
    print(f"🤖 Embedding: all-MiniLM-L6-v2 (lazy loaded)")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get('/api/debug/assets')
async def debug_assets():
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'images')
    files = []
    if os.path.exists(assets_dir):
        for root, dirs, filenames in os.walk(assets_dir):
            for f in filenames:
                files.append(os.path.relpath(os.path.join(root, f), assets_dir))
    return {'assets_found': len(files), 'sample_files': files[:20]}
