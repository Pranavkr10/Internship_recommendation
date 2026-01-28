const resumeInput = document.getElementById("resumeFile");
const resultsDiv = document.getElementById("results");

resumeInput.addEventListener("change", async () => {
  if (!resumeInput.files.length) return;

  resultsDiv.innerHTML = "<p>Processing resume…</p>";

  const formData = new FormData();
  formData.append("resume", resumeInput.files[0]);

  try {
    // 1) Upload + parse resume
    const uploadRes = await fetch("/api/resume/upload", {
      method: "POST",
      body: formData
    });

    const uploadData = await uploadRes.json();

    if (!uploadRes.ok || !uploadData.success) {
      resultsDiv.innerHTML = `<p style="color:red">${uploadData.error || "Upload failed"}</p>`;
      return;
    }

    const resume = uploadData.parsed_resume;
    const ats = uploadData.ats_score;

    const skills = Array.isArray(resume.skills) && resume.skills.length
      ? resume.skills.join(", ")
      : "No skills detected";

    // ✅ show upload result immediately
    resultsDiv.innerHTML = `
      <h3>Resume Parsed Successfully</h3>
      <p><b>Skills:</b> ${skills}</p>
      <p><b>ATS Score:</b> ${ats.ats_score} (${ats.grade})</p>

      <h4>Top Improvements</h4>
      <ul>
        ${Array.isArray(ats.improvements) ? ats.improvements.map(i => `<li>${i}</li>`).join("") : ""}
      </ul>

      <hr/>
      <p>Finding recommended internships...</p>
    `;

    // 2) Get recommendations from DB/CSV
    const recRes = await fetch("/api/resume/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resume_text: uploadData.text_preview ? uploadData.text_preview : "",
        top_n: 5,
        use_db: true
      })
    });

    const recData = await recRes.json();

    if (!recRes.ok || !recData.success) {
      resultsDiv.innerHTML += `<p style="color:red">Recommendation failed: ${recData.error || "Unknown error"}</p>`;
      return;
    }

    // ✅ render recommended companies
    const companies = Array.isArray(recData.recommended_companies)
      ? recData.recommended_companies
      : [];

    const internships = Array.isArray(recData.recommendations)
      ? recData.recommendations
      : [];

    resultsDiv.innerHTML += `
      <h3>Recommended Companies</h3>
      ${
        companies.length
          ? `<ul>${companies.map(c => `<li><b>${c.name}</b> (${c.count})</li>`).join("")}</ul>`
          : "<p>No companies found.</p>"
      }

      <h3>Top Internship Matches</h3>
      ${
        internships.length
          ? internships.map(r => `
              <div style="border:1px solid #ddd;padding:10px;margin:10px 0;border-radius:8px;">
                <p><b>${r.internship.title}</b> - ${r.internship.company}</p>
                <p><b>Score:</b> ${Number(r.score).toFixed(3)}</p>
                <p><b>Skills Needed:</b> ${r.internship.required_skills || "N/A"}</p>
              </div>
            `).join("")
          : "<p>No internship recommendations.</p>"
      }

      <p style="font-size:12px;color:gray;">
        Data source: ${recData.data_source}
      </p>
    `;

  } catch (err) {
    console.error(err);
    resultsDiv.innerHTML = "<p style='color:red'>Server error</p>";
  }
});
