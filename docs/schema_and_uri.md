# KyungHee-Chatbot — URI 규칙 & 메타데이터 스키마 (v1.0)

본 문서는 규정/세칙 PDF에서 생성된 청크 및 인덱스 메타데이터에 적용되는 영구 식별자 규칙(URN/HTTP)과 메타 스키마를 정의합니다.

## 1) 안정식 식별자 (URN/HTTP)

### 1.1 URN (내부 기준/역참조)

* **형태:** `urn:khu:reg:{code}:{versionDate}:art{N}[:cl{M}]`
- `code`: 규정 식별자 (예: AA, GS)
- `versionDate`: 시행일 (YYYY-MM-DD)
- `artN`: 제 N조
- `clM`: (선택) 제 M항

* **예시:** `urn:khu:reg:AA:2024-09-01:art15:cl2`

### 1.2 HTTP(S) 영구 URI (외부 노출/SPARQL 탐색)

* **조 레벨 형태:** `https://kg.khu.ac.kr/reg/{code}-{versionDate}#art{N}`

* **항 레벨 형태:** `https://kg.khu.ac.kr/reg/{code}-{versionDate}#art{N}-cl{M}`

* **예시:** `https://kg.khu.ac.kr/reg/AA-2024-09-01#art15-cl2`

**운영 방침:** `uri` (URN)와 `articleUri/clauseUri` (HTTP)를 병행 저장합니다.

---

## 2) 메타 스키마 정의 (v1.0)

| 필드명 | 타입 및 제약 조건 | 비고 |
| ----- | ----- | ----- |
| `schema_version` | `"1.0"` (string) | **필수**. 현재 스키마 버전. |
| `uri` | string(URN) | **필수**. URN 형태의 영구 식별자. |
| `articleUri` | string(HTTP URI) \| null | **권장**. 조(Article) 레벨의 HTTP URI. |
| `clauseUri` | string(HTTP URI) \| null | **권장**. 항(Clause) 레벨의 HTTP URI. |
| `documentCode` | string | 규정 식별자 (예: AA, GS). |
| `versionDate` | YYYY-MM-DD (date string) | **필수**. 규정의 시행일. |
| `effectiveFrom` | YYYY-MM-DD \| null | 효력 시작일. 없으면 `null`. |
| `effectiveUntil` | YYYY-MM-DD \| null | 효력 종료일. 없으면 `null`. |
| `contentType` | enum{text\|table\|annex\|appendix} | **필수**. 콘텐츠 유형. |
| `articleNumber` | int | 조 번호. |
| `clauseNumber` | int \| null | 항 번호. 항이 없으면 `null`. |
| `program` | enum{UG,MS,PHD,IME_MS,...} \| null | 적용 프로그램. 대문자/언더스코어 표준. 없으면 `null`. |
| `cohort` | enum{Cohort_2022,Cohort_2023,...} \| null | 적용 기수. 형식: `Cohort_YYYY`. 없으면 `null`. |
| `sourceFile` | string(filename) \| null | 원본 파일명. 생성 불가 시 `null`. |
| `page` | int \| null | 원본 PDF의 페이지 번호 (1-based). 없으면 `null`. |
| `md5` | string \| null | 청크 텍스트 기준 MD5 해시값. 없으면 `null`. |
| `overrides` | string\[\] (uri\[\]) | 이 조항이 무효화/대체하는 이전 조항의 URI 목록. |
| `cites` | string\[\] (uri\[\]) | 이 조항이 참조하는 다른 조항의 URI 목록. |
| `hasExceptionFor` | (string\|uri)\[\] | 예외가 적용되는 대상 목록 (자유 텍스트/URI 혼용). |