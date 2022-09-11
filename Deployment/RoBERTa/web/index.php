<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <meta name="description" content="" />
    <meta name="author" content="" />
    <title>RoBERTa</title>
    <!-- Favicon-->
    <link rel="icon" type="image/x-icon" href="assets/favicon.ico" />
    <!-- Font Awesome icons (free version)-->
    <script src="https://use.fontawesome.com/releases/v6.1.0/js/all.js" crossorigin="anonymous"></script>
    <!-- Google fonts-->
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css" />
    <link href="https://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic" rel="stylesheet"
        type="text/css" />
    <!-- Core theme CSS (includes Bootstrap)-->
    <link href="css/styles.css" rel="stylesheet" />
    <script src="./js/predict.js"></script>
</head>

<body id="page-top">
    <!-- Navigation-->
    <nav class="navbar navbar-expand-lg bg-secondary text-uppercase fixed-top" id="mainNav">
        <div class="container">
            <button class="navbar-toggler text-uppercase font-weight-bold bg-primary text-white rounded" type="button"
                data-bs-toggle="collapse" data-bs-target="#navbarResponsive" aria-controls="navbarResponsive"
                aria-expanded="false" aria-label="Toggle navigation">
                Menu
                <i class="fas fa-bars"></i>
            </button>
            <div class="collapse navbar-collapse" id="navbarResponsive">
                <ul class="navbar-nav m-auto">
                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded"
                            href="#main">Home</a></li>
                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded"
                            href="#try">Try it!</a></li>
                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded"
                            href="#download">Download thesis</a></li>
                </ul>
            </div>
        </div>
    </nav>
    <!-- Masthead-->
    <header class="masthead bg-primary text-white text-center" id="main">
        <div class="container d-flex align-items-center flex-column">
            <!-- Masthead Avatar Image-->
            <img class="masthead-avatar mb-5" src="assets/img/news.svg" alt="..." />
            <!-- Masthead Heading-->
            <h1 class="masthead-heading text-uppercase mb-0">Neural news classifier</h1>
            <!-- Masthead Subheading-->
            <p class="masthead-subheading font-weight-light mb-0">Deployed by Adrián Vázquez Barrera </p>
        </div>
    </header>
    <!-- Portfolio Section-->
    <section class="page-section portfolio" id="try">
        <div class="container">
            <!-- Portfolio Section Heading-->
            <h2 class="page-section-heading text-center text-secondary mb-0">RoBERTa - News Classifier
            </h2>
            <!-- Icon Divider-->
            <div class="divider-custom">
                <div class="divider-custom-line"></div>
                <div class="divider-custom-icon"><i class="fas fa-star"></i></div>
                <div class="divider-custom-line"></div>
            </div>
            <!-- Portfolio Grid Items-->
            <form action="http://localhost:8080" method="post" enctype='multipart/form-data'target="_blank">
                <div class="row justify-content-center">
                    <textarea name="text" id="text" cols="30" rows="3" placeholder="Write here"
                        class="display-5"></textarea>
                    <div class="text-center">
                        <button class="btn btn-primary btn-lg mt-5" type="submmit">Guess Tag</button>
                    </div>
                </div>
            </form>
        </div>
    </section>
    <!-- About Section-->
    <section class="page-section bg-primary text-white mb-0" id="download">
        <div class="container">
            <!-- About Section Heading-->
            <h2 class="page-section-heading text-center text-uppercase text-white mb-5">Download thesis</h2>
            <!-- About Section Button-->
            <div class="text-center mt-5">
                <a class="btn btn-xl btn-outline-light" href="THESIS.pdf">
                    <i class="fas fa-download me-2"></i>
                    Download
                </a>
            </div>
        </div>
    </section>
    <!-- Footer-->
    <footer>
        <!-- Copyright Section-->
        <div class="copyright py-4 text-center text-white">
            <div class="container"><small>&copy; Adrián Vázquez Barrera</small></div>
        </div>
    </footer>
    <!-- Bootstrap core JS-->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <!-- Core theme JS-->
    <script src="js/scripts.js"></script>
    <script src="https://cdn.startbootstrap.com/sb-forms-latest.js"></script>
</body>

</html>